extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Attribute, Expr, FnArg, ItemFn, Lit, Meta, Pat, ReturnType, Type};

/// Extract the doc comments on the original function, concatenated and trimmed.
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

/// Parse the tool attribute arguments for error strategy and retries.
fn parse_tool_config(args: TokenStream) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    let default_error_strategy =
        quote! { transformers::pipelines::text_generation::ErrorStrategy::Fail };
    let default_retries = quote! { 3u32 };

    if args.is_empty() {
        return (default_error_strategy, default_retries);
    }

    let mut error_strategy = default_error_strategy;
    let mut retries = default_retries;

    // Try to parse as multiple comma-separated arguments
    let args_str = args.to_string();

    // Split by comma and process each part
    for part in args_str.split(',') {
        let part = part.trim();

        if part.starts_with("on_error") {
            // Extract the value after the =
            if let Some(value_part) = part.split('=').nth(1) {
                let value_part = value_part.trim();
                if let Ok(expr) = syn::parse_str::<syn::Expr>(value_part) {
                    error_strategy = parse_error_strategy_from_expr(&expr);
                }
            }
        } else if part.starts_with("retries") {
            // Extract the value after the =
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
    // Convert the expression to a string to check what it contains
    let expr_str = quote!(#expr).to_string();

    // Clean up the string (remove extra spaces)
    let expr_str = expr_str.replace(" ", "");

    // Handle different forms of the error strategy
    if expr_str == "Fail" || expr_str.contains("ErrorStrategy::Fail") {
        quote! { transformers::pipelines::text_generation::ErrorStrategy::Fail }
    } else if expr_str == "ReturnToModel" || expr_str.contains("ErrorStrategy::ReturnToModel") {
        quote! { transformers::pipelines::text_generation::ErrorStrategy::ReturnToModel }
    } else {
        // Generate a compile-time error for invalid strategies
        syn::Error::new_spanned(
            expr,
            "Unknown error strategy. Valid options are: ErrorStrategy::Fail, ErrorStrategy::ReturnToModel"
        ).to_compile_error()
    }
}

/// Check if the function returns a Result type
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

/// Attribute macro `#[tool]` that turns a nice Rust function into a `Tool`.
///
/// Example:
/// ```
/// use transformers::tool;
///
/// #[tool]
/// fn add(a: i32, b: i32) -> String {
///     (a + b).to_string()
/// }
///
/// #[tool(on_error = ErrorStrategy::ReturnToModel)]
/// fn get_weather(city: String) -> Result<String, WeatherError> {
///     // ...
/// }
///
/// // In user code:
/// let mut pipeline = ...;
/// pipeline.register_tools(tools![add]);
/// ```
#[proc_macro_attribute]
pub fn tool(args: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the error strategy and retries from args
    let (error_strategy, max_retries) = parse_tool_config(args);

    // Parse the source function.
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name_ident = &input_fn.sig.ident;
    let fn_name_str = fn_name_ident.to_string();
    let wrapper_name = format_ident!("__{}_tool_wrapper", fn_name_ident);
    let tool_builder_name = format_ident!("__{}_tool_builder", fn_name_ident);
    let params_struct_name = format_ident!("__{}_ToolParams", fn_name_ident);
    let schema_fn_name = format_ident!("__{}_tool_schema", fn_name_ident);

    // Doc comments become description.
    let description = extract_doc(&input_fn.attrs);

    // Check if function returns Result
    let is_result = returns_result(&input_fn.sig.output);

    // Gather parameter information.
    let mut param_fields = Vec::new();
    let mut param_idents = Vec::new();

    // Traverse function arguments.
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

    // Generate different wrapper logic based on return type
    let wrapper_body = if is_result {
        quote! {
            let parsed: #params_struct_name = serde_json::from_value(parameters)
                .map_err(|e| transformers::pipelines::text_generation::ToolError::Format(e.to_string()))?;
            let #params_struct_name { #( #param_idents ),* } = parsed;
            use transformers::pipelines::text_generation::ToolError;
            let result = #fn_name_ident( #(#param_idents),* );

            // Convert the result to the expected type
            match result {
                Ok(s) => Ok(s),
                Err(e) => Err(ToolError::Message(e.to_string())),
            }
        }
    } else {
        quote! {
            let parsed: #params_struct_name = serde_json::from_value(parameters)
                .map_err(|e| transformers::pipelines::text_generation::ToolError::Format(e.to_string()))?;
            let #params_struct_name { #( #param_idents ),* } = parsed;
            let result = #fn_name_ident( #(#param_idents),* );
            Ok(result)
        }
    };

    // Generate the output tokens: keep original fn, plus wrapper + data.
    let expanded = quote! {
        // Keep the original function as-is
        #input_fn

        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        struct #params_struct_name {
            #( #param_fields ),*
        }

        // Automatically generated wrapper that matches the `Tool` function signature.
        #[doc(hidden)]
        fn #wrapper_name(parameters: serde_json::Value) -> Result<String, transformers::pipelines::text_generation::ToolError> {
            #wrapper_body
        }

        #[doc(hidden)]
        fn #schema_fn_name() -> schemars::schema::RootSchema {
            schemars::schema_for!(#params_struct_name)
        }

        // Hidden function used by the tools! macro
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

        // Generate a module with the same name as the function
        // This allows the tools! macro to find the builder
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
