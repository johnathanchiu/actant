/**
 * Bun preload — registers happy-dom as the global DOM so React
 * components can mount under @testing-library/react in unit tests.
 *
 * Also wires `cleanup()` from @testing-library/react to run after every
 * test so DOM state doesn't leak across renders.
 *
 * Referenced from `bunfig.toml` (preload).
 */
import { GlobalRegistrator } from '@happy-dom/global-registrator'

if (typeof window === 'undefined') {
  GlobalRegistrator.register()
}

import { afterEach } from 'bun:test'
import { cleanup } from '@testing-library/react'

afterEach(() => {
  cleanup()
})
