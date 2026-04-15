use std::future::Future;
use std::marker::PhantomData;

use async_trait::async_trait;
use claumini_core::{Tool, ToolContext, ToolDescriptor, ToolError};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{ToolMetadata, ToolRegistration};

pub struct SyncToolAdapter<I, O, F> {
    metadata: ToolMetadata,
    handler: F,
    _marker: PhantomData<fn(I) -> O>,
}

pub struct AsyncToolAdapter<R, F> {
    handler: F,
    _marker: PhantomData<fn() -> R>,
}

impl<R, F> AsyncToolAdapter<R, F> {
    pub fn new(handler: F) -> Self {
        Self {
            handler,
            _marker: PhantomData,
        }
    }
}

impl<I, O, F> SyncToolAdapter<I, O, F> {
    pub fn new(metadata: ToolMetadata, handler: F) -> Self {
        Self {
            metadata,
            handler,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<I, O, F> Tool for SyncToolAdapter<I, O, F>
where
    I: Send + Sync + Serialize + DeserializeOwned + JsonSchema + 'static,
    O: Send + Sync + Serialize + DeserializeOwned + JsonSchema + 'static,
    F: Fn(I) -> Result<O, ToolError> + Send + Sync,
{
    type Input = I;
    type Output = O;

    fn descriptor(&self) -> ToolDescriptor {
        self.metadata.descriptor_for::<I, O>()
    }

    async fn call(
        &self,
        input: Self::Input,
        _ctx: &mut ToolContext,
    ) -> Result<Self::Output, ToolError> {
        (self.handler)(input)
    }
}

#[async_trait]
impl<R, F, Fut> Tool for AsyncToolAdapter<R, F>
where
    R: ToolRegistration,
    F: Fn(R::Input) -> Fut + Send + Sync,
    Fut: Future<Output = Result<R::Output, ToolError>> + Send,
{
    type Input = R::Input;
    type Output = R::Output;

    fn descriptor(&self) -> ToolDescriptor {
        R::tool_descriptor()
    }

    async fn call(
        &self,
        input: Self::Input,
        _ctx: &mut ToolContext,
    ) -> Result<Self::Output, ToolError> {
        (self.handler)(input).await
    }
}
