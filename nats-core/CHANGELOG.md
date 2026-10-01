# Unreleased

### Added

- `ConnectionClosedError`: in-flight `request()` waiters are now failed immediately with this error when the connection closes permanently (explicit close, reconnection disabled, or reconnect attempts exhausted), instead of receiving a bare `asyncio.CancelledError` or waiting out the request timeout. Transient disconnects followed by a successful reconnect leave the waiter pending, and a response racing the close completes the waiter exactly once.

# 0.2.0

### Added

- `Client.rtt()` method to measure round-trip time to the server (#859)

### Changed

- Request/reply multiplexed over a single inbox subscription instead of creating a new subscription per request (#825)
- Build backend from setuptools to uv_build (#813)

### Fixed

- Various type annotation issues in client and protocol handling (#827)

# 0.1.0

Initial release
