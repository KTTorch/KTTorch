package io.github.kt.torch.nn

typealias Block = ai.djl.nn.Block
typealias Blocks = ai.djl.nn.Blocks
typealias Activation = ai.djl.nn.Activation
typealias BlockFactory = ai.djl.nn.BlockFactory
typealias BlockList = ai.djl.nn.BlockList
typealias IdentityBlockFactory = ai.djl.nn.IdentityBlockFactory
typealias LambdaBlock = ai.djl.nn.LambdaBlock
typealias OnesBlockFactory = ai.djl.nn.OnesBlockFactory
typealias ParallelBlock = ai.djl.nn.ParallelBlock
typealias Parameter = ai.djl.nn.Parameter
typealias ParameterBuilder = ai.djl.nn.Parameter.Builder
typealias ParameterList = ai.djl.nn.ParameterList
typealias SequentialBlock = ai.djl.nn.SequentialBlock
typealias SymbolBlock = ai.djl.nn.SymbolBlock
typealias UninitializedParameterException = ai.djl.nn.UninitializedParameterException

@DslMarker
annotation class LayerFactoryDsl

@LayerFactoryDsl
class LayersBuilder {
    internal val layers = mutableListOf<Block>()

    operator fun Block.unaryPlus() {
        layers.add(this)
    }
}