use std::any::Any;
use std::collections::hash_map::RandomState;
use std::ops::Add;

use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
use polars_arrow::export::arrow::array::PrimitiveArray;
use polars_arrow::export::arrow::compute::aggregate::Sum;
use polars_arrow::export::arrow::types::simd::Simd;
use polars_core::export::num::NumCast;
use polars_core::prelude::*;
use polars_core::utils::arrow::compute::aggregate::sum_primitive;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;
use crate::operators::{ArrowDataType, IdxSize};

#[derive(Clone)]
pub struct ApproxCountAgg {
    hllp: HyperLogLogPlus<AnyValue<'static>, RandomState>,
}

impl ApproxCountAgg {
    pub(crate) fn new(precision: u8) -> Self {
        let estimator = HyperLogLogPlus::new(precision, RandomState::new())
            .expect("Unable to create cardinality estimator.");
        ApproxCountAgg { hllp: estimator }
    }
    fn get_count(&mut self) -> IdxSize {
        self.hllp.count().trunc() as IdxSize
    }
}

impl AggregateFn for ApproxCountAgg {
    fn has_physical_agg(&self) -> bool {
        true
    }

    fn pre_agg(&mut self, _chunk_idx: IdxSize, item: &mut dyn ExactSizeIterator<Item = AnyValue>) {
        let item = unsafe { item.next().unwrap_unchecked_release() };
        self.hllp
            .insert(unsafe { &item.into_static().unwrap_unchecked() });
    }

    // TODO recheck
    fn pre_agg_ordered(
        &mut self,
        _chunk_idx: IdxSize,
        offset: IdxSize,
        length: IdxSize,
        values: &Series,
    ) {
    }

    fn dtype(&self) -> DataType {
        IDX_DTYPE
    }

    fn combine(&mut self, other: &dyn Any) {
        let other = unsafe { other.downcast_ref::<Self>().unwrap_unchecked_release() };
        self.hllp
            .merge(&other.hllp)
            .expect("estimators should be mergable.");
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        AnyValue::from(self.get_count())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
