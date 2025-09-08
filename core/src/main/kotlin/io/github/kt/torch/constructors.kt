package io.github.kt.torch

import io.github.kt.torch.tensor.*
import io.github.kt.torch.ndarray.types.*
import java.io.*
import java.nio.*
import java.nio.charset.*
import java.nio.file.*

fun KTTorch.tensor(shape: Shape) =
    Tensors.manager().create(shape)

fun KTTorch.tensor(data: Number) =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: Float) =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: Int) =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: Double) =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: Long) =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: Byte) =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: Boolean) =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: String) =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: Array<String>) =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: Array<String>, charset: Charset) =
    Tensors.manager().create(data, charset)

fun KTTorch.tensor(data: Array<String>, shape: Shape) =
    Tensors.manager().create(data, shape)

fun KTTorch.tensor(data: Array<String>, charset: Charset, shape: Shape) =
    Tensors.manager().create(data, charset, shape)

fun KTTorch.tensor(data: FloatArray): Tensor =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: IntArray): Tensor =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: DoubleArray): Tensor =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: LongArray): Tensor =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: ByteArray): Tensor =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: BooleanArray): Tensor =
    Tensors.manager().create(data)

fun KTTorch.tensor(data: Buffer, shape: Shape): Tensor =
    Tensors.manager().create(data, shape)

fun KTTorch.tensor(shape: Shape, dataType: DataType): Tensor =
    Tensors.manager().create(shape, dataType)

fun KTTorch.tensor(data: Buffer, shape: Shape, dataType: DataType): Tensor =
    Tensors.manager().create(data, shape, dataType)

fun KTTorch.tensor(data: FloatArray, shape: Shape): Tensor =
    Tensors.manager().create(data, shape)

fun KTTorch.tensor(data: IntArray, shape: Shape): Tensor =
    Tensors.manager().create(data, shape)

fun KTTorch.tensor(data: DoubleArray, shape: Shape): Tensor =
    Tensors.manager().create(data, shape)

fun KTTorch.tensor(data: LongArray, shape: Shape): Tensor =
    Tensors.manager().create(data, shape)

fun KTTorch.tensor(data: ByteArray, shape: Shape): Tensor =
    Tensors.manager().create(data, shape)

fun KTTorch.tensor(data: BooleanArray, shape: Shape): Tensor =
    Tensors.manager().create(data, shape)

fun KTTorch.tensorCSR(data: FloatArray, indptr: LongArray, indices: LongArray, shape: Shape, device: Device): Tensor =
    Tensors.manager().createCSR(data, indptr, indices, shape, device)

fun KTTorch.tensorCSR(data: Buffer, indptr: LongArray, indices: LongArray, shape: Shape, device: Device): Tensor =
    Tensors.manager().createCSR(data, indptr, indices, shape, device)

fun KTTorch.tensorCSR(data: Buffer, indptr: LongArray, indices: LongArray, shape: Shape): Tensor =
    Tensors.manager().createCSR(data, indptr, indices, shape)

fun KTTorch.tensorRowSparse(data: Buffer, dataShape: Shape, indices: LongArray, shape: Shape, device: Device): Tensor =
    Tensors.manager().createRowSparse(data, dataShape, indices, shape, device)

fun KTTorch.tensorRowSparse(data: Buffer, dataShape: Shape, indices: LongArray, shape: Shape): Tensor =
    Tensors.manager().createRowSparse(data, dataShape, indices, shape)

fun KTTorch.tensorCoo(data: Buffer, indices: Array<LongArray>, shape: Shape) =
    Tensors.manager().createCoo(data, indices, shape)

fun KTTorch.decode(bytes: ByteArray): Tensor =
    Tensors.manager().decode(bytes)

fun KTTorch.decode(stream: InputStream): Tensor =
    Tensors.manager().decode(stream)

fun KTTorch.load(path: Path): TensorList =
    Tensors.manager().load(path)

fun KTTorch.load(path: Path, device: Device): TensorList =
    Tensors.manager().load(path, device)

fun KTTorch.zeros(shape: Shape): Tensor =
    Tensors.manager().zeros(shape)

fun KTTorch.zeros(shape: Shape, dataType: DataType): Tensor =
    Tensors.manager().zeros(shape, dataType)

fun KTTorch.zeros(shape: Shape, dataType: DataType, device: Device): Tensor =
    Tensors.manager().zeros(shape, dataType, device)

fun KTTorch.arange(stop: Int): Tensor =
    Tensors.manager().arange(stop)

fun KTTorch.arange(stop: Float): Tensor =
    Tensors.manager().arange(stop)

fun KTTorch.arange(start: Int, stop: Int): Tensor =
    Tensors.manager().arange(start, stop)

fun KTTorch.arange(start: Float, stop: Float): Tensor =
    Tensors.manager().arange(start, stop)

fun KTTorch.arange(start: Int, stop: Int, step: Int): Tensor =
    Tensors.manager().arange(start, stop, step)

fun KTTorch.arange(start: Float, stop: Float, step: Float): Tensor =
    Tensors.manager().arange(start, stop, step)

fun KTTorch.arange(start: Int, stop: Int, step: Int, dataType: DataType): Tensor =
    Tensors.manager().arange(start, stop, step, dataType)

fun KTTorch.arange(start: Float, stop: Float, step: Float, dataType: DataType): Tensor =
    Tensors.manager().arange(start, stop, step, dataType)

fun KTTorch.arange(start: Float, stop: Float, step: Float, dataType: DataType, device: Device): Tensor =
    Tensors.manager().arange(start, stop, step, dataType, device)

fun KTTorch.eye(rows: Int): Tensor =
    Tensors.manager().eye(rows)

fun KTTorch.eye(rows: Int, k: Int): Tensor =
    Tensors.manager().eye(rows, k)

fun KTTorch.eye(rows: Int, cols: Int, k: Int): Tensor =
    Tensors.manager().eye(rows, cols, k)

fun KTTorch.eye(rows: Int, cols: Int, k: Int, dataType: DataType): Tensor =
    Tensors.manager().eye(rows, cols, k, dataType)

fun KTTorch.eye(rows: Int, cols: Int, k: Int, dataType: DataType, device: Device): Tensor =
    Tensors.manager().eye(rows, cols, k, dataType, device)

fun KTTorch.full(shape: Shape, fillValue: Int): Tensor =
    Tensors.manager().full(shape, fillValue)

fun KTTorch.full(shape: Shape, fillValue: Float): Tensor =
    Tensors.manager().full(shape, fillValue)

fun KTTorch.full(shape: Shape, fillValue: Float, dType: DataType): Tensor =
    Tensors.manager().full(shape, fillValue, dType)

fun KTTorch.full(shape: Shape, fillValue: Float, dType: DataType, device: Device): Tensor =
    Tensors.manager().full(shape, fillValue, dType, device)

fun KTTorch.hanningWindow(numPoints: Long): Tensor =
    Tensors.manager().hanningWindow(numPoints)

fun KTTorch.linspace(start: Int, stop: Int, num: Int): Tensor =
    Tensors.manager().linspace(start, stop, num)

fun KTTorch.linspace(start: Float, stop: Float, num: Int): Tensor =
    Tensors.manager().linspace(start, stop, num)

fun KTTorch.linspace(start: Int, stop: Int, num: Int, endpoint: Boolean): Tensor =
    Tensors.manager().linspace(start, stop, num, endpoint)

fun KTTorch.linspace(start: Float, stop: Float, num: Int, endpoint: Boolean): Tensor =
    Tensors.manager().linspace(start, stop, num, endpoint)

fun KTTorch.linspace(start: Float, stop: Float, num: Int, endpoint: Boolean, device: Device): Tensor =
    Tensors.manager().linspace(start, stop, num, endpoint, device)

fun KTTorch.ones(shape: Shape): Tensor =
    Tensors.manager().ones(shape)

fun KTTorch.ones(shape: Shape, dataType: DataType): Tensor =
    Tensors.manager().ones(shape, dataType)

fun KTTorch.ones(shape: Shape, dataType: DataType, device: Device): Tensor =
    Tensors.manager().ones(shape, dataType, device)

fun KTTorch.randomInteger(low: Long, high: Long, shape: Shape, dataType: DataType) =
    Tensors.manager().randomInteger(low, high, shape, dataType)

fun KTTorch.randomPermutation(n: Long) =
    Tensors.manager().randomPermutation(n)

fun KTTorch.randomUniform(low: Float, high: Float, shape: Shape) =
    Tensors.manager().randomUniform(low, high, shape)

fun KTTorch.randomUniform(low: Float, high: Float, shape: Shape, dataType: DataType) =
    Tensors.manager().randomUniform(low, high, shape, dataType)

fun KTTorch.randomUniform(low: Float, high: Float, shape: Shape, dataType: DataType, device: Device) =
    Tensors.manager().randomUniform(low, high, shape, dataType, device)

fun KTTorch.randomNormal(shape: Shape): Tensor =
    Tensors.manager().randomNormal(shape)

fun KTTorch.randomNormal(shape: Shape, dataType: DataType): Tensor =
    Tensors.manager().randomNormal(shape, dataType)

fun KTTorch.randomNormal(loc: Float, scale: Float, shape: Shape, dataType: DataType): Tensor =
    Tensors.manager().randomNormal(loc, scale, shape, dataType)

fun KTTorch.randomNormal(loc: Float, scale: Float, shape: Shape, dataType: DataType, device: Device): Tensor =
    Tensors.manager().randomNormal(loc, scale, shape, dataType, device)

fun KTTorch.truncatedNormal(shape: Shape) =
    Tensors.manager().truncatedNormal(shape)

fun KTTorch.truncatedNormal(shape: Shape, dataType: DataType) =
    Tensors.manager().truncatedNormal(shape, dataType)

fun KTTorch.truncatedNormal(loc: Float, scale: Float, shape: Shape, dataType: DataType) =
    Tensors.manager().truncatedNormal(loc, scale, shape, dataType)

fun KTTorch.truncatedNormal(loc: Float, scale: Float, shape: Shape, dataType: DataType, device: Device) =
    Tensors.manager().truncatedNormal(loc, scale, shape, dataType, device)

fun KTTorch.randomMultinomial(n: Int, pValues: Tensor): Tensor =
    Tensors.manager().randomMultinomial(n, pValues)

fun KTTorch.randomMultinomial(n: Int, pValues: Tensor, shape: Shape): Tensor =
    Tensors.manager().randomMultinomial(n, pValues, shape)

fun KTTorch.sampleNormal(mu: Tensor, sigma: Tensor) =
    Tensors.manager().sampleNormal(mu, sigma)

fun KTTorch.sampleNormal(mu: Tensor, sigma: Tensor, shape: Shape) =
    Tensors.manager().sampleNormal(mu, sigma, shape)

fun KTTorch.samplePoisson(lambda: Tensor) =
    Tensors.manager().samplePoisson(lambda)

fun KTTorch.samplePoisson(lambda: Tensor, shape: Shape) =
    Tensors.manager().samplePoisson(lambda, shape)

fun KTTorch.sampleGamma(alpha: Tensor, beta: Tensor) =
    Tensors.manager().sampleGamma(alpha, beta)

fun KTTorch.sampleGamma(alpha: Tensor, beta: Tensor, shape: Shape) =
    Tensors.manager().sampleGamma(alpha, beta, shape)
