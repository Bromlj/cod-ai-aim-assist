# ScreenSight 0.2 engine test build

This test release packages the previously saved v0.2 tracking, time-based movement,
fresh-frame input checks, and detector-buffer changes with the existing v0.1
interface. The larger blocked interface update is NOT included or republished.

## Important differences from the earlier v0.2 proposal

- The existing window may still display ScreenSight 0.1. The engine version is
  0.2.0; identify this executable using its build commit and SHA-256, not the
  old interface title.
- No new preset controls, preference-saving controls or diagnostic-export buttons
  have been added. The backend supports additional options but this retained
  interface does not expose them.
- The retained interface still enables its optional overlay by default. Disable
  **Show click-through overlay** before starting. The user observed pointer/focus
  problems with the old overlay in Black Ops 7. Exclusive-fullscreen compatibility
  has not been established.
- Mouse control still starts disabled. F8 disables mouse control. Nothing sends
  clicks or keyboard presses. There are no new drivers or anti-cheat bypasses.
- Last performance metrics can remain visible while waiting for window focus;
  they are not live measurements when the status says WAITING or READY.

## Test scope and limitations

The unit tests exercise geometric association, confirmation after a detection gap,
frame-rate-normalized movement in an idealized simulation, direction reversal,
stale-frame rejection, and post-inference focus/button checks. The Windows UI
smoke test checks the retained interface rather than the unpublished redesign.

These are not gameplay accuracy tests. The detector has not been retrained;
it still cannot distinguish teams, and mouse counts are not game-camera angles.
No claim is made that moving-target tracking is fixed in Black Ops 7. This build
is for evaluation of the candidate changes, not a validated final gaming product.

An antivirus result only applies to the exact SHA-256 of the scanned executable.
No detection by an antivirus engine is not a guarantee of safety or game approval.
