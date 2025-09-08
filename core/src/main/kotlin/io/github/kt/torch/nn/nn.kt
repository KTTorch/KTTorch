package io.github.kt.torch.nn

import io.github.kt.torch.ndarray.NDList
import io.github.kt.torch.ndarray.types.Shape
import io.github.kt.torch.nn.convolutional.*
import io.github.kt.torch.nn.norm.*
import io.github.kt.torch.nn.pooling.*
import io.github.kt.torch.nn.recurrent.*
import io.github.kt.torch.nn.transformer.*
import io.github.kt.torch.nn.transformer.BertMaskedLanguageModelLoss as BertMaskedLanguageModelLossImpl
import io.github.kt.torch.nn.transformer.BertNextSentenceLoss as BertNextSentenceLossImpl
import io.github.kt.torch.nn.transformer.BertPretrainingLoss as BertPretrainingLossImpl
import io.github.kt.torch.nn.core.*
import io.github.kt.torch.nn.recurrent.GRUBuilder
import io.github.kt.torch.nn.transformer.BertBlockBuilder
import io.github.kt.torch.tensor.*
import io.github.kt.torch.training.loss.Loss
import io.github.kt.torch.nn.core.ConstantEmbedding as ConstantEmbeddingModule
import io.github.kt.torch.nn.core.SparseMax as SparseMaxModule

object nn {

    fun ReLU(): Block = Activation.reluBlock()
    fun ReLU6(): Block = Activation.relu6Block()
    fun Sigmoid(): Block = Activation.sigmoidBlock()
    fun Tanh(): Block = Activation.tanhBlock()
    fun SoftPlus(): Block = Activation.softPlusBlock()
    fun SoftSign(): Block = Activation.softSignBlock()
    fun LeakyReLU(alpha: Float): Block = Activation.leakyReluBlock(alpha)
    fun ELU(alpha: Float): Block = Activation.eluBlock(alpha)
    fun SeLU(): Block = Activation.seluBlock()
    fun GeLU(): Block = Activation.geluBlock()
    fun Swish(beta: Float): Block = Activation.swishBlock(beta)
    fun Mish(): Block = Activation.mishBlock()
    fun PreLU(): Block = Activation.preluBlock()

    fun Parallel(f: (List<NDList>) -> NDList, g: LayersBuilder.() -> Unit): Block {
        val b = LayersBuilder().apply(g)
        return ParallelBlock(f).apply { b.layers.forEach(::add)}
    }

    fun Parameter(f: ParameterBuilder.() -> Unit): Parameter =
        Parameter.builder().apply(f).build()

    fun Sequential(f: LayersBuilder.() -> Unit): Block {
        val b = LayersBuilder().apply(f)
        return SequentialBlock().apply { b.layers.forEach(::add)}
    }

    fun Conv1d(f: Conv1dBuilder.() -> Unit): Block =
        Conv1d.builder().apply(f).build()

    fun Conv1dTranspose(f: Conv1dTransposeBuilder.() -> Unit): Block =
        Conv1dTranspose.builder().apply(f).build()

    fun Conv2d(f: Conv2dBuilder.() -> Unit): Block =
        Conv2d.builder().apply(f).build()

    fun Conv2dTranspose(f: Conv2dTransposeBuilder.() -> Unit): Block =
        Conv2dTranspose.builder().apply(f).build()

    fun Conv3d(f: Conv3dBuilder.() -> Unit): Block =
        Conv3d.builder().apply(f).build()

    fun ConstantEmbedding(embedding: Tensor): Block = ConstantEmbeddingModule(embedding)

    fun Linear(f: LinearBuilder.() -> Unit): Block = Linear.builder().apply(f).build()

    fun LinearCollection(f: LinearCollectionBuilder.() -> Unit): Block =
        LinearCollection.builder().apply(f).build()

    fun Multiplication(f: MultiplicationBuilder.() -> Unit): Block =
        Multiplication.builder().apply(f).build()

    fun SparseMax(): Block = SparseMaxModule()
    fun SparseMax(axis: Int): Block = SparseMaxModule(axis)
    fun SparseMax(axis: Int, topK: Int): Block = SparseMaxModule(axis, topK)

    fun BatchNorm(f: BatchNormBuilder.() -> Unit): Block {
        val builder: BatchNormBuilder = BatchNorm.builder() as BatchNormBuilder
        return builder.apply(f).build()
    }

    fun Dropout(f: DropoutBuilder.() -> Unit): Block =
        Dropout.builder().apply(f).build()
    fun GhostBatchNorm(f: GhostBatchNormBuilder.() -> Unit): Block =
        GhostBatchNorm.builder().apply(f).build()
    fun LayerNorm(f: LayerNormBuilder.() -> Unit): Block =
        LayerNorm.builder().apply(f).build()

    fun MaxPool1d(kernelShape: Shape): Block =
        Pool.maxPool1dBlock(kernelShape)
    fun MaxPool1d(kernelShape: Shape, stride: Shape): Block =
        Pool.maxPool1dBlock(kernelShape, stride)
    fun MaxPool1d(kernelShape: Shape, stride: Shape, padding: Shape): Block =
        Pool.maxPool1dBlock(kernelShape, stride, padding)
    fun MaxPool1d(kernelShape: Shape, stride: Shape, padding: Shape, ceilMode: Boolean): Block =
        Pool.maxPool1dBlock(kernelShape, stride, padding, ceilMode)
    fun MaxPool2d(kernelShape: Shape): Block =
        Pool.maxPool2dBlock(kernelShape)
    fun MaxPool2d(kernelShape: Shape, stride: Shape): Block =
        Pool.maxPool2dBlock(kernelShape, stride)
    fun MaxPool2d(kernelShape: Shape, stride: Shape, padding: Shape): Block =
        Pool.maxPool2dBlock(kernelShape, stride, padding)
    fun MaxPool2d(kernelShape: Shape, stride: Shape, padding: Shape, ceilMode: Boolean): Block =
        Pool.maxPool2dBlock(kernelShape, stride, padding, ceilMode)
    fun MaxPool3d(kernelShape: Shape, stride: Shape): Block =
        Pool.maxPool3dBlock(kernelShape, stride)
    fun MaxPool3d(kernelShape: Shape, stride: Shape, padding: Shape): Block =
        Pool.maxPool3dBlock(kernelShape, stride, padding)
    fun MaxPool3d(kernelShape: Shape, stride: Shape, padding: Shape, ceilMode: Boolean): Block =
        Pool.maxPool3dBlock(kernelShape, stride, padding, ceilMode)
    fun GlobalMaxPool1d(): Block = Pool.globalMaxPool1dBlock()
    fun GlobalMaxPool2d(): Block = Pool.globalMaxPool2dBlock()
    fun GlobalMaxPool3d(): Block = Pool.globalMaxPool3dBlock()
    fun GlobalAvgPool1d(): Block = Pool.globalAvgPool1dBlock()
    fun GlobalAvgPool2d(): Block = Pool.globalAvgPool2dBlock()
    fun GlobalAvgPool3d(): Block = Pool.globalAvgPool3dBlock()
    fun AvgPool1d(kernelShape: Shape): Block =
        Pool.avgPool1dBlock(kernelShape)
    fun AvgPool1d(kernelShape: Shape, stride: Shape): Block =
        Pool.avgPool1dBlock(kernelShape, stride)
    fun AvgPool1d(kernelShape: Shape, stride: Shape, padding: Shape): Block =
        Pool.avgPool1dBlock(kernelShape, stride, padding)
    fun AvgPool1d(kernelShape: Shape, stride: Shape, padding: Shape, ceilMode: Boolean): Block =
        Pool.avgPool1dBlock(kernelShape, stride, padding, ceilMode)
    fun AvgPool2d(kernelShape: Shape): Block =
        Pool.avgPool2dBlock(kernelShape)
    fun AvgPool2d(kernelShape: Shape, stride: Shape): Block =
        Pool.avgPool2dBlock(kernelShape, stride)
    fun AvgPool2d(kernelShape: Shape, stride: Shape, padding: Shape): Block =
        Pool.avgPool2dBlock(kernelShape, stride, padding)
    fun AvgPool2d(kernelShape: Shape, stride: Shape, padding: Shape, ceilMode: Boolean): Block =
        Pool.avgPool2dBlock(kernelShape, stride, padding, ceilMode)
    fun AvgPool3d(kernelShape: Shape): Block =
        Pool.avgPool3dBlock(kernelShape)
    fun AvgPool3d(kernelShape: Shape, stride: Shape): Block =
        Pool.avgPool3dBlock(kernelShape, stride)
    fun AvgPool3d(kernelShape: Shape, stride: Shape, padding: Shape): Block =
        Pool.avgPool3dBlock(kernelShape, stride, padding)
    fun AvgPool3d(kernelShape: Shape, stride: Shape, padding: Shape, ceilMode: Boolean): Block =
        Pool.avgPool3dBlock(kernelShape, stride, padding, ceilMode)
    fun LpPool1d(normType: Float, kernelShape: Shape): Block =
        Pool.lpPool1dBlock(normType, kernelShape)
    fun LpPool1d(normType: Float, kernelShape: Shape, stride: Shape, padding: Shape): Block =
        Pool.lpPool1dBlock(normType, kernelShape, stride, padding)
    fun LpPool1d(normType: Float, kernelShape: Shape, stride: Shape, padding: Shape, ceilMode: Boolean): Block =
        Pool.lpPool1dBlock(normType, kernelShape, stride, padding, ceilMode)
    fun LpPool2d(normType: Float, kernelShape: Shape): Block =
        Pool.lpPool2dBlock(normType, kernelShape)
    fun LpPool2d(normType: Float, kernelShape: Shape, stride: Shape, padding: Shape): Block =
        Pool.lpPool2dBlock(normType, kernelShape, stride, padding)
    fun LpPool2d(normType: Float, kernelShape: Shape, stride: Shape, padding: Shape, ceilMode: Boolean): Block =
        Pool.lpPool2dBlock(normType, kernelShape, stride, padding, ceilMode)
    fun LpPool3d(normType: Float, kernelShape: Shape): Block =
        Pool.lpPool3dBlock(normType, kernelShape)
    fun LpPool3d(normType: Float, kernelShape: Shape, stride: Shape, padding: Shape): Block =
        Pool.lpPool3dBlock(normType, kernelShape, stride, padding)
    fun LpPool3d(normType: Float, kernelShape: Shape, stride: Shape, padding: Shape, ceilMode: Boolean): Block =
        Pool.lpPool3dBlock(normType, kernelShape, stride, padding, ceilMode)
    fun GlobalLpPool1d(normType: Float): Block = Pool.globalLpPool1dBlock(normType)
    fun GlobalLpPool2d(normType: Float): Block = Pool.globalLpPool2dBlock(normType)
    fun GlobalLpPool3d(normType: Float): Block = Pool.globalLpPool3dBlock(normType)

    fun GRU(f: GRUBuilder.() -> Unit): Block = GRU.builder().apply(f).build()
    fun LSTM(f: LSTMBuilder.() -> Unit): Block = LSTM.builder().apply(f).build()
    fun RNN(f: RNNBuilder.() -> Unit): Block = RNN.builder().apply(f).build()

    fun Bert(f: BertBlockBuilder.() -> Unit): BertBlock =
        BertBlock.builder().apply(f).build()
    fun BertMaskedLanguageModel(bertBlock: BertBlock, hiddenActivation: (Tensor) -> Tensor): Block =
        BertMaskedLanguageModelBlock(bertBlock, hiddenActivation)
    fun BertMaskedLanguageModelLoss(labelIdx: Int, maskIdx: Int, logProbsIdx: Int): Loss = BertMaskedLanguageModelLossImpl(labelIdx, maskIdx, logProbsIdx)
    fun BertNextSentence(): Block = BertNextSentenceBlock()
    fun BertNextSentenceLoss(labelIdx: Int, nextSentencePredictionIdx: Int): Loss = BertNextSentenceLossImpl(labelIdx, nextSentencePredictionIdx)
    fun BertPretraining(f: BertBlockBuilder.() -> Unit): Block = BertPretrainingBlock(BertBlock.builder().apply(f))
    fun BertPretrainingLoss(): Loss = BertPretrainingLossImpl()
    fun IdEmbedding(f: IdEmbeddingBuilder.() -> Unit): Block = IdEmbeddingBuilder().apply(f).build()
    fun PointwiseFeedForward(hiddenSizes: List<Int>, outputSize: Int, activationFunction: (NDList) -> NDList): Block =
        PointwiseFeedForwardBlock(hiddenSizes, outputSize, activationFunction)
    fun ScaledDotProductAttention(f: ScaledDotProductAttentionBlockBuilder.() -> Unit): Block =
        ScaledDotProductAttentionBlock.builder().apply(f).build()
    fun TransformerEncoder(embeddingSize: Int, headCount: Int, hiddenSize: Int, dropoutProbability: Float, activationFunction: (NDList) -> NDList): Block =
        TransformerEncoderBlock(embeddingSize, headCount, hiddenSize, dropoutProbability, activationFunction)
}