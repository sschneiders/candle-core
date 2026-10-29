//! MAI FORK regression: detach_keep keeps downstream ops recorded while
//! cutting the gradient path at the boundary — the block-local training
//! arm's exact requirement (session 28).

use candle_core::{DType, Device, Tensor, Var};

#[test]
fn detach_keep_boundary() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let w1 = Var::ones((2, 2), DType::F32, &dev)?;
    let w2 = Var::ones((2, 2), DType::F32, &dev)?;
    let x = Tensor::ones((2, 2), DType::F32, &dev)?;
    let y = w1.as_tensor().mul(&x)?;
    let b = y.detach_keep();
    let z = b.matmul(w2.as_tensor())?.sum_all()?;
    let grads = z.backward()?;
    assert!(grads.get(w1.as_tensor()).is_none(), "grad crossed the boundary");
    assert!(grads.get(w2.as_tensor()).is_some(), "downstream op not recorded");
    // plain detach still erases the downstream recording for ALL-non-var args
    let y2 = w1.as_tensor().mul(&x)?;
    let b2 = y2.detach();
    let c2 = b2.affine(2.0, 0.0)?; // all args non-variable → not recorded
    let z2 = c2.sum_all()?;
    let g = z2.backward()?;
    assert!(g.get(w1.as_tensor()).is_none());
    Ok(())
}
