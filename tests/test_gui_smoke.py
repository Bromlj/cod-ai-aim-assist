"""Windows compatibility test for the retained v0.1 UI with the v0.2 engine.

The larger interface redesign is not included in this test release. This test
checks the interface that is actually shipped, not unimplemented preset buttons.
"""
import sys
import pytest

@pytest.mark.skipif(sys.platform != 'win32', reason='Windows Tk overlay attributes')
def test_retained_interface_starts_disarmed_and_accepts_engine_options():
    from screensight.gui import MainWindow
    from screensight.settings import VERSION
    window = MainWindow()
    try:
        window.root.update_idletasks()
        window.root.update()
        window._poll()
        assert VERSION == '0.2.0'
        assert window.root.title().startswith('ScreenSight')
        assert not window.enabled.get()
        assert not window.engine.snapshot().input_enabled
        # Overlay is explicitly disabled for this compatibility check. The old
        # interface still defaults it on; release notes warn users to turn it off.
        window.show_overlay.set(False)
        window._configure()
        window.root.update_idletasks()
        assert window.engine.options().overlay is False
        assert not window.overlay._shown
        assert window.engine.options().deadzone == 2.0
        assert window.engine.options().lead_ms == 0.0
    finally:
        window.close()
