package io.github.kt.torch.tensor

typealias Tensor = io.github.kt.torch.ndarray.NDArray
typealias TensorList = io.github.kt.torch.ndarray.NDList

operator fun Tensor.plus(other: Tensor): Tensor = this.add(other)
operator fun Tensor.plus(n: Number): Tensor = this.add(n)
operator fun Tensor.plusAssign(other: Tensor) {
    this.addi(other)
}
operator fun Tensor.plusAssign(n: Number) {
    this.addi(n)
}
operator fun Tensor.minus(other: Tensor): Tensor = this.sub(other)
operator fun Tensor.minus(n: Number): Tensor = this.sub(n)
operator fun Tensor.minusAssign(other: Tensor) {
    this.subi(other)
}
operator fun Tensor.minusAssign(n: Number) {
    this.subi(n)
}
operator fun Tensor.times(other: Tensor): Tensor = this.mul(other)
operator fun Tensor.times(n: Number): Tensor = this.mul(n)
operator fun Tensor.timesAssign(other: Tensor) {
    this.muli(other)
}
operator fun Tensor.timesAssign(n: Number) {
    this.muli(n)
}
operator fun Tensor.divAssign(other: Tensor) {
    this.divi(other)
}
operator fun Tensor.divAssign(n: Number) {
    this.divi(n)
}
operator fun Tensor.rem(other: Tensor): Tensor = this.mod(other)
operator fun Tensor.rem(n: Number): Tensor = this.mod(n)
operator fun Tensor.remAssign(other: Tensor) {
    this.modi(other)
}
operator fun Tensor.remAssign(n: Number) {
    this.modi(n)
}
operator fun Tensor.inc(): Tensor = this.addi(1f)
operator fun Tensor.dec(): Tensor = this.subi(1f)
operator fun Tensor.set(index: Tensor, value: Tensor): Tensor = this.put(index, value)
operator fun Tensor.compareTo(other: Tensor): Int = this.eq(other).argMax().getLong().toInt()
operator fun Tensor.compareTo(n: Number): Int = this.eq(n).argMax().getLong().toInt()
operator fun Tensor.not(): Tensor = this.neg()
