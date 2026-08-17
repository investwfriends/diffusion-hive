# Mzinga engine binaries

The native Mzinga C# engine (the UHP teacher) is required for data generation
and evaluation. Binaries are platform-specific and kept out of git:

- **Linux x64**: auto-downloaded by `setup_cloud.sh` (also settable via
  `MZINGA_ENGINE_PATH`). Source: https://github.com/jonthysell/Mzinga/releases
- **macOS arm64**: download the same release and place the binary at
  `Mzinga.MacOSArm64/MzingaEngine`.

Set `MZINGA_ENGINE_PATH` to override the binary location. The engine needs
`DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1` on minimal images (missing libicu).
