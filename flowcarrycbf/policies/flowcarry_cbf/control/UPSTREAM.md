# Algorithm provenance

- The Conditional Flow Matching objective and Temporal U-Net structure are adapted through
  `flowcarrycbf.policies.safe_flow.model` from the vendored `SafeFlowMatcher-main` reference.
- Receding-horizon direct action generation, full-sequence candidate sampling, and first-action
  execution follow the design pattern in the vendored `SafeFlowMPC-main` reference.
- V2 does not copy SafeFlowMPC's seven-axis robot model or Acados controller. Tiago V2 uses its
  own 17-dimensional action contract, full Tiago Dual Pinocchio URDF, and OSQP safety programs.

The corresponding third-party source trees and their licenses remain under `third_party/`.
