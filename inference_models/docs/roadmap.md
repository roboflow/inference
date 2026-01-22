## 🛣️ Roadmap to Stable Release

We're actively working toward stabilizing `inference-models` and integrating it into the main `inference` package. The plan is to:

1. **Stabilize the API** - Finalize the core interfaces and ensure backward compatibility
2. **Integrate with `inference`** - Make `inference-models` available as a selectable backend in the `inference` package
3. **Production deployment** - Enable users to choose between the classic inference backend and the new `inference-models` backend
4. **Gradual migration** - Provide a smooth transition path for existing users

We're sharing this preview to gather valuable community feedback that will help us shape the final release. Your input is crucial in making this the best inference experience possible!

!!! note "Current Status"
    The `inference-models` package is approaching stability but is still in active development.

    * The core API is stabilizing, but minor changes may still occur
    * We're working toward backward compatibility guarantees
    * Production use is possible but we recommend thorough testing
    * For mission-critical systems, continue using the stable `inference` package until the official integration is complete

