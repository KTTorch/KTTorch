package io.github.kt.torch.tensor

import io.github.kt.torch.Device
import io.github.kt.torch.ndarray.*

internal object Tensors {
    internal val rootMgr: NDManager by lazy {
        NDManager.newBaseManager().also { r ->
            Runtime.getRuntime().addShutdownHook(Thread { r.close() })
        }
    }

    private val stackTL = ThreadLocal.withInitial { ArrayDeque<NDManager>() }

    inline fun <T> withScope(block: () -> T): T {
        val parent = stackTL.get().lastOrNull() ?: rootMgr
        val sub = parent.newSubManager()
        stackTL.get().addLast(sub)
        try {
            return block()
        } finally {
            stackTL.get().removeLast()
            sub.close()
        }
    }

    inline fun <T> withScope(device: Device, block: () -> T): T {
        val parent = stackTL.get().lastOrNull() ?: rootMgr
        val sub = parent.newSubManager(device)
        stackTL.get().addLast(sub)
        try {
            return block()
        } finally {
            stackTL.get().removeLast()
            sub.close()
        }
    }

    inline fun <T> withScope(mgr: NDManager, block: () -> T): T {
        stackTL.get().addLast(mgr)
        try {
            return block()
        } finally {
            stackTL.get().removeLast()
        }
    }

    internal fun manager(): NDManager =
        stackTL.get().lastOrNull() ?: error("No NDManager bound to current thread. Use torch.scope { ... } or torch.scope(ctx.ndManager) { ... }")
}