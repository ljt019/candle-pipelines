extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Attribute, Expr, FnArg, ItemFn, Lit, Meta, Pat, ReturnType, Type};

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

fn parse_tool_config(args: TokenStream) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    let default_error_strategy =
        quote! { transformers::pipelines::text_generation::ErrorStrategy::Fail };
    let default_retries = quote! { 3u32 };

    if args.is_empty() {
        return (default_error_strategy, default_retries);
    }

    let mut error_strategy = default_error_strategy;
    let mut retries = default_retries;

    let args_str = args.to_string();

    for part in args_str.split(',') {
        let part = part.trim();

        if part.starts_with("on_error") {
            if let Some(value_part) = part.split('=').nth(1) {
                let value_part = value_part.trim();
                if let Ok(expr) = syn::parse_str::<syn::Expr>(value_part) {
                    error_strategy = parse_error_strategy_from_expr(&expr);
                }
            }
        } else if part.starts_with("retries") {
            if let Some(value_part) = part.split('=').nth(1) {
                let value_part = value_part.trim();
                if let Ok(lit) = syn::parse_str::<syn::LitInt>(value_part) {
                    let retry_count = lit.base10_parse::<u32>().unwrap_or(3);
                    retries = quote! { #retry_count };
                }
            }
        }
    }

    (error_strategy, retries)
}

fn parse_error_strategy_from_expr(expr: &syn::Expr) -> proc_macro2::TokenStream {
    let expr_str = quote!(#expr).to_string();

    let expr_str = expr_str.replace(" ", "");

    if expr_str == "Fail" || expr_str.contains("ErrorStrategy::Fail") {
        quote! { transformers::pipelines::text_generation::ErrorStrategy::Fail }
    } else if expr_str == "ReturnToModel" || expr_str.contains("ErrorStrategy::ReturnToModel") {
        quote! { transformers::pipelines::text_generation::ErrorStrategy::ReturnToModel }
    } else {
        syn::Error::new_spanned(
            expr,
            "Unknown error strategy. Valid options are: ErrorStrategy::Fail, ErrorStrategy::ReturnToModel"
        ).to_compile_error()
    }
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

#[proc_macro_attribute]
pub fn tool(args: TokenStream, item: TokenStream) -> TokenStream {
    let (error_strategy, max_retries) = parse_tool_config(args);

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

    for arg in input_fn.sig.inputs.iter() {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = &pat_ident.ident;
                let ty = &*pat_type.ty;

                param_fields.push(quote! { pub #param_name: #ty });
                param_idents.push(quote! { #param_name });
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
                    .map_err(|e| transformers::TransformersError::Tool(
                        transformers::error::ToolError::InvalidParams {
                            name: #fn_name_str.to_string(),
                            reason: e.to_string(),
                        }
                    ))?;
                let #params_struct_name { #( #param_idents ),* } = parsed;
                let result = #call_invocation;

                match result {
                    Ok(s) => Ok(s),
                    Err(e) => Err(transformers::TransformersError::Tool(
                        transformers::error::ToolError::ExecutionFailed {
                            name: #fn_name_str.to_string(),
                            attempts: 1,
                            reason: e.to_string(),
                        }
                    )),
                }
            })
        }
    } else {
        quote! {
            Box::pin(async move {
                let parsed: #params_struct_name = serde_json::from_value(parameters)
                    .map_err(|e| transformers::TransformersError::Tool(
                        transformers::error::ToolError::InvalidParams {
                            name: #fn_name_str.to_string(),
                            reason: e.to_string(),
                        }
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
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        struct #params_struct_name {
            #( #param_fields ),*
        }

        #[doc(hidden)]
        fn #wrapper_name(parameters: serde_json::Value) -> transformers::pipelines::text_generation::ToolFuture {
            #wrapper_body
        }

        #[doc(hidden)]
        fn #schema_fn_name() -> schemars::schema::RootSchema {
            schemars::schema_for!(#params_struct_name)
        }

        #[doc(hidden)]
        pub fn #tool_builder_name() -> transformers::pipelines::text_generation::Tool {
            let schema = #schema_fn_name();

            transformers::pipelines::text_generation::Tool::new(
                #fn_name_str.to_string(),
                #description.to_string(),
                schema,
                #wrapper_name,
                #error_strategy,
                #max_retries,
            )
        }

        #[doc(hidden)]
        pub mod #fn_name_ident {
            use super::*;

            pub fn __tool() -> transformers::pipelines::text_generation::Tool {
                #tool_builder_name()
            }
        }
    };

    TokenStream::from(expanded)
}
