"""Repository-local IsaacLab compatibility helpers."""

from __future__ import annotations


def ensure_contact_data_buffer(max_contact_data_count: int = 64) -> bool:
    """Enable raw contact-point buffers on IsaacLab 2.2 / Isaac Sim 4.5.

    IsaacLab 2.3 exposes this as ``ContactSensorCfg.max_contact_data_count``.
    Version 2.2 creates every rigid-contact view with the PhysX default of zero,
    so ``get_contact_data`` fails.  The XHand front-pad observation requires the
    raw points. ``max_contact_data_count`` is the total capacity of one contact
    view, not a per-environment value, so callers must scale it with the number
    of environments. Patching the view factory is process-local and leaves the
    external IsaacLab installation untouched.

    Returns ``True`` when the compatibility wrapper was installed.
    """
    from omni.physics.tensors.impl.api import SimulationView

    original = SimulationView.create_rigid_contact_view
    if getattr(original, "_dexgen_contact_buffer_patch", False):
        return False

    def create_rigid_contact_view_with_buffer(self, *args, **kwargs):
        kwargs.setdefault("max_contact_data_count", max_contact_data_count)
        return original(self, *args, **kwargs)

    create_rigid_contact_view_with_buffer._dexgen_contact_buffer_patch = True
    SimulationView.create_rigid_contact_view = create_rigid_contact_view_with_buffer
    return True
