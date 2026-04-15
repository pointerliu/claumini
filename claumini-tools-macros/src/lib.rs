#![forbid(unsafe_code)]

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Data, DeriveInput, Fields, Ident, LitStr, Result, Token, Type, parse_macro_input};

#[proc_macro_derive(ToolRegistration, attributes(tool))]
pub fn derive_tool_registration(input: TokenStream) -> TokenStream {
    match derive_tool_registration_impl(parse_macro_input!(input as DeriveInput)) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn derive_tool_registration_impl(input: DeriveInput) -> Result<proc_macro2::TokenStream> {
    ensure_zero_sized_marker(&input)?;

    let args = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("tool"))
        .ok_or_else(|| syn::Error::new_spanned(&input.ident, "missing #[tool(...)] attribute"))?
        .parse_args::<ToolArgs>()?;

    let ident = input.ident;
    let name = args.name;
    let description = args.description;
    let input_ty = args.input;
    let output_ty = args.output;

    Ok(quote! {
        impl ::claumini_tools::ToolRegistration for #ident {
            type Input = #input_ty;
            type Output = #output_ty;

            fn tool_metadata() -> ::claumini_tools::ToolMetadata {
                ::claumini_tools::ToolMetadata::new(#name, #description)
            }
        }
    })
}

fn ensure_zero_sized_marker(input: &DeriveInput) -> Result<()> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "ToolRegistration can only be derived for structs",
        ));
    };

    match &data.fields {
        Fields::Unit => Ok(()),
        Fields::Named(fields) if fields.named.is_empty() => Ok(()),
        Fields::Unnamed(fields) if fields.unnamed.is_empty() => Ok(()),
        _ => Err(syn::Error::new_spanned(
            &input.ident,
            "ToolRegistration requires a zero-sized marker struct",
        )),
    }
}

struct ToolArgs {
    name: LitStr,
    description: LitStr,
    input: Type,
    output: Type,
}

impl Parse for ToolArgs {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let args = Punctuated::<ToolArg, Token![,]>::parse_terminated(input)?;

        let mut name = None;
        let mut description = None;
        let mut input_ty = None;
        let mut output_ty = None;

        for arg in args {
            match arg {
                ToolArg::Name(value) => assign_once(&mut name, value, "name")?,
                ToolArg::Description(value) => assign_once(&mut description, value, "description")?,
                ToolArg::Input(value) => assign_once(&mut input_ty, value, "input")?,
                ToolArg::Output(value) => assign_once(&mut output_ty, value, "output")?,
            }
        }

        Ok(Self {
            name: name.ok_or_else(|| syn::Error::new(input.span(), "missing tool name"))?,
            description: description
                .ok_or_else(|| syn::Error::new(input.span(), "missing tool description"))?,
            input: input_ty.ok_or_else(|| syn::Error::new(input.span(), "missing input type"))?,
            output: output_ty
                .ok_or_else(|| syn::Error::new(input.span(), "missing output type"))?,
        })
    }
}

fn assign_once<T>(slot: &mut Option<T>, value: T, field: &str) -> Result<()> {
    if slot.replace(value).is_some() {
        Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("duplicate tool {field}"),
        ))
    } else {
        Ok(())
    }
}

enum ToolArg {
    Name(LitStr),
    Description(LitStr),
    Input(Type),
    Output(Type),
}

impl Parse for ToolArg {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let ident = input.parse::<Ident>()?;
        input.parse::<Token![=]>()?;

        if ident == "name" {
            Ok(Self::Name(input.parse()?))
        } else if ident == "description" {
            Ok(Self::Description(input.parse()?))
        } else if ident == "input" {
            Ok(Self::Input(input.parse()?))
        } else if ident == "output" {
            Ok(Self::Output(input.parse()?))
        } else {
            Err(syn::Error::new_spanned(
                ident,
                "unsupported tool argument; expected one of: name, description, input, output",
            ))
        }
    }
}
