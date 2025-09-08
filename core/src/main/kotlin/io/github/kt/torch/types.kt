package io.github.kt.torch

typealias Application = ai.djl.Application
typealias BaseModel = ai.djl.BaseModel
typealias Device = ai.djl.Device
typealias MalformedModelException = ai.djl.MalformedModelException
typealias Model = ai.djl.Model
typealias ModelException = ai.djl.ModelException
typealias TrainingDivergedException = ai.djl.TrainingDivergedException
typealias torch = KTTorch
typealias kt = KTTorch

class Devices {
    companion object {
        val GPU = Device.gpu()
        val CPU = Device.cpu()
        val MPS = Device.of("mps", 0)
    }
}
