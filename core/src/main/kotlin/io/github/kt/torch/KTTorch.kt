package io.github.kt.torch

import io.github.kt.torch.ndarray.NDManager
import io.github.kt.torch.tensor.*

object KTTorch {
    fun scope(block: () -> Unit) = Tensors.withScope(block)

    fun scope(device: Device, block: () -> Unit) = Tensors.withScope(device, block)

    fun scope(mgr: NDManager, block: () -> Unit) = Tensors.withScope(mgr, block)

    operator fun <T> get(vararg elements: T): List<T> = elements.toList()
}