//! Procedural macros for defining tools that can be called by language models.
//!
//! See [`tool`] and [`tools!`] for usage.

extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Attribute, Expr, FnArg, Ident, ItemFn, Lit, Meta, Pat, ReturnType, Token, Type,
};

fn extract_doc(attrs: &[Attribute]) -> String {
    let mut out = String::new();
    for attr in attrs {
        if let Meta::NameValue(nv) = &attr.meta {
            if nv.path.is_ident("doc") {
                if let Expr::Lit(expr_lit) = &nv.value {
                    if let Lit::Str(lit_str) = &expr_lit.lit {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(lit_str.value().trim());
                    }
                }
            }
        }
    }
    out
}

fn parse_tool_config(args: TokenStream) -> proc_macro2::TokenStream {
    let default_retries = quote! { 3u32 };

    if args.is_empty() {
        return default_retries;
    }

    let args_str = args.to_string();

    for part in args_str.split(',') {
        let part = part.trim();

        if part.starts_with("retries") {
            if let Some(value_part) = part.split('=').nth(1) {
                let value_part = value_part.trim();
                if let Ok(lit) = syn::parse_str::<syn::LitInt>(value_part) {
                    let retry_count = lit.base10_parse::<u32>().unwrap_or(3);
                    return quote! { #retry_count };
                }
            }
        }
    }

    default_retries
}

fn returns_result(output: &ReturnType) -> bool {
    if let ReturnType::Type(_, ty) = output {
        if let Type::Path(type_path) = &**ty {
            if let Some(segment) = type_path.path.segments.last() {
                return segment.ident == "Result";
            }
        }
    }
    false
}

/// Converts a function into a tool the model can call.
///
/// The function's doc comment becomes the tool description shown to the model.
/// Parameters are automatically converted to a JSON schema.
///
/// # Example
///
/// ```rust,ignore
/// use candle_pipelines::text_generation::tool;
/// use candle_pipelines::error::Result;
///
/// #[tool]
/// /// Get the current weather for a city.
/// fn get_weather(city: String) -> Result<String> {
///     Ok(format!("Weather in {}: sunny", city))
/// }
/// ```
///
/// # Options
///
/// - `retries = N` - Max retry attempts if tool fails (default: 3)
///
/// ```rust,ignore
/// #[tool(retries = 5)]
/// fn flaky_tool(x: i32) -> Result<String> { Ok("done".into()) }
/// ```
#[proc_macro_attribute]
pub fn tool(args: TokenStream, item: TokenStream) -> TokenStream {
    let max_retries = parse_tool_config(args);

    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name_ident = &input_fn.sig.ident;
    let fn_name_str = fn_name_ident.to_string();
    let wrapper_name = format_ident!("__{}_tool_wrapper", fn_name_ident);
    let tool_builder_name = format_ident!("__{}_tool_builder", fn_name_ident);
    let params_struct_name = format_ident!("__{}_ToolParams", fn_name_ident);
    let schema_fn_name = format_ident!("__{}_tool_schema", fn_name_ident);
    let is_async = input_fn.sig.asyncness.is_some();

    let description = extract_doc(&input_fn.attrs);

    let is_result = returns_result(&input_fn.sig.output);

    let mut param_fields = Vec::new();
    let mut param_idents = Vec::new();
    let mut param_types = Vec::new();

    for arg in input_fn.sig.inputs.iter() {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = &pat_ident.ident;
                let ty = &*pat_type.ty;

                param_fields.push(quote! { pub #param_name: #ty });
                param_idents.push(quote! { #param_name });
                param_types.push(quote! { #ty });
            }
        }
    }

    let call_invocation = if is_async {
        quote! { #fn_name_ident( #(#param_idents),* ).await }
    } else {
        quote! { #fn_name_ident( #(#param_idents),* ) }
    };

    let wrapper_body = if is_result {
        quote! {
            Box::pin(async move {
                let parsed: #params_struct_name = serde_json::from_value(parameters)
                    .map_err(|e| candle_pipelines::error::PipelineError::Tool(
                        format!("Invalid parameters for '{}': {}", #fn_name_str, e)
                    ))?;
                let #params_struct_name { #( #param_idents ),* } = parsed;
                let result = #call_invocation;

                match result {
                    Ok(s) => Ok(s),
                    Err(e) => Err(candle_pipelines::error::PipelineError::Tool(
                        format!("Tool '{}' failed: {}", #fn_name_str, e)
                    )),
                }
            })
        }
    } else {
        quote! {
            Box::pin(async move {
                let parsed: #params_struct_name = serde_json::from_value(parameters)
                    .map_err(|e| candle_pipelines::error::PipelineError::Tool(
                        format!("Invalid parameters for '{}': {}", #fn_name_str, e)
                    ))?;
                let #params_struct_name { #( #param_idents ),* } = parsed;
                let result = #call_invocation;
                Ok(result)
            })
        }
    };

    let expanded = quote! {
        #input_fn

        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #[derive(serde::Deserialize)]
        struct #params_struct_name {
            #( #param_fields ),*
        }

        impl candle_pipelines::text_generation::schemars::JsonSchema for #params_struct_name {
            fn schema_name() -> String {
                stringify!(#params_struct_name).to_string()
            }

            fn json_schema(gen: &mut candle_pipelines::text_generation::schemars::gen::SchemaGenerator) -> candle_pipelines::text_generation::schemars::schema::Schema {
                let mut schema_object = candle_pipelines::text_generation::schemars::schema::SchemaObject {
                    instance_type: Some(candle_pipelines::text_generation::schemars::schema::InstanceType::Object.into()),
                    ..Default::default()
                };
                let mut properties = candle_pipelines::text_generation::schemars::Map::new();
                let mut required = std::collections::BTreeSet::new();

                #(
                    properties.insert(
                        stringify!(#param_idents).to_string(),
                        gen.subschema_for::<#param_types>(),
                    );
                    required.insert(stringify!(#param_idents).to_string());
                )*

                schema_object.object = Some(Box::new(candle_pipelines::text_generation::schemars::schema::ObjectValidation {
                    properties,
                    required,
                    ..Default::default()
                }));

                candle_pipelines::text_generation::schemars::schema::Schema::Object(schema_object)
            }
        }

        #[doc(hidden)]
        fn #wrapper_name(parameters: serde_json::Value) -> candle_pipelines::text_generation::ToolFuture {
            #wrapper_body
        }

        #[doc(hidden)]
        fn #schema_fn_name() -> candle_pipelines::text_generation::schemars::schema::RootSchema {
            let gen = candle_pipelines::text_generation::schemars::gen::SchemaGenerator::default();
            gen.into_root_schema_for::<#params_struct_name>()
        }

        #[doc(hidden)]
        pub fn #tool_builder_name() -> candle_pipelines::text_generation::Tool {
            let schema = #schema_fn_name();

            candle_pipelines::text_generation::Tool::new(
                #fn_name_str.to_string(),
                #description.to_string(),
                schema,
                #wrapper_name,
                #max_retries,
            )
        }

        #[doc(hidden)]
        pub mod #fn_name_ident {
            use super::*;

            pub fn __tool() -> candle_pipelines::text_generation::Tool {
                #tool_builder_name()
            }
        }
    };

    TokenStream::from(expanded)
}

struct ToolsList {
    tools: Punctuated<Ident, Token![,]>,
}

impl Parse for ToolsList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(ToolsList {
            tools: Punctuated::parse_terminated(input)?,
        })
    }
}

/// Collects multiple tools into a `Vec<Tool>` for registration.
///
/// # Example
///
/// ```rust,ignore
/// use candle_pipelines::text_generation::{tools, tool, TextGenerationPipeline, Qwen3};
/// use candle_pipelines::error::Result;
///
/// #[tool]
/// /// Get weather.
/// fn get_weather(city: String) -> Result<String> { Ok(city) }
///
/// async fn example(pipeline: TextGenerationPipeline<Qwen3>) {
///     pipeline.register_tools(tools![get_weather]).await;
/// }
/// ```
#[proc_macro]
pub fn tools(input: TokenStream) -> TokenStream {
    let ToolsList { tools } = parse_macro_input!(input as ToolsList);

    let tool_calls = tools.iter().map(|ident| {
        quote! { #ident::__tool() }
    });

    let expanded = quote! {
        vec![#(#tool_calls),*]
    };

    TokenStream::from(expanded)
}
