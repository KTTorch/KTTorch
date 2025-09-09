# KTTorch

**KTTorch** is a Kotlin deep learning library **inspired by PyTorch**.  
It aims to bring a similar API style and development experience into the Kotlin ecosystem.

Currently **KTTorch** is in **alpha** and is not ready for production use.  It is essentially a proof of concept where 
syntactical sugar is added to the DJL API (defaulted to the PyTorch-Engine) to make it more Kotlin-like. 

## Examples of the API:
```kotlin
val inputSize = (28 * 28).toLong()
val outputSize = 10L
val mlp = nn.Sequential {
    +nn.BatchFlatten(inputSize)
    +nn.Linear {
        setUnits(128)
    }
    +nn.ReLU()
    +nn.Linear {
        setUnits(64)
    }
    +nn.ReLU()
    +nn.Linear {
        setUnits(outputSize)
    }
}

torch.scope {
    val a = torch.tensor(floatArrayOf(1f, 2f, 3f, 4f))
    val b = torch.tensor(floatArrayOf(5f, 6f, 7f, 8f))
    val c = a + b + 3
    assertContentEquals(floatArrayOf(9f, 11f, 13f, 15f), c.toFloatArray())
    val index = torch.tensor(0)
    c[index] = 10f
    val s = torch.tensor(2)
    println(c.reshape(Shape(1, 4)).shape)
    println(s.shape)
    a.pow(4)
}
```
## Reasoning and Design

Function extensions have been added to the object KTTorch to make the API more Kotlin-like and to resemble PyTorch.
The entire DJL API has been typealiased so the the API appears to reside within the KTTorch namespace but compile to the 
DJL API. BUT the added extensions remain in the KTTorch namespace and provide a more Kotlin-like API.

The ```torch.scope``` function is used to create a new scope where a ThreadLocal NDManager is created ensuring that all 
operations are performed within the scope.  This is to ensure that all operations are performed within the same 
NDManager and therefore share the same NDArrays.  This is necessary to ensure that the NDArrays are not garbage collected 
before they are used. Furthermore, the scope is closed when the scope is exited - cleaning up the NDManager and all ```Tensors```
created within the scope.  An alternative ```torch.scope(manager)``` function is provided to allow the user to specify a 
custom NDManager for use where a context provides an NDManager tailored to the scope.

The ```nn.Sequential``` function is used to create a sequential block of layers.  The ```+``` operator is used to add 
layers to the block.  All DJL Builders are used appropiately and Kotlin turns it into a form of DSL that reads a lot like
configuration rather than code. 

---

⚠️ **Disclaimer**:  
KTTorch is an **independent open-source project** and has **no affiliation with Jetbrains, the PyTorch Foundation, the Linux Foundation, or Meta AI**.

---
