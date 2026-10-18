#!/bin/bash

# ==========================================
# SolTTS macOS DMG packaging script
# Assembles publish-mac-* directories into an .app bundle and packages it as a .dmg installer
#
# Usage:
#   ./pack-mac-dmg.sh [VERSION] [OUTPUT_DIR]
#   Defaults: VERSION=v1.2.2, OUTPUT_DIR=releases
# ==========================================

set -e

VERSION="${1:-v1.2.2}"
OUTPUT_DIR="${2:-releases}"
APP_NAME="SolTTS"
BUNDLE_ID="com.soltts.app"

mkdir -p "$OUTPUT_DIR"

# Ensure required commands exist
if ! command -v hdiutil &> /dev/null; then
    echo "❌ hdiutil is unavailable on this system (.dmg can only be created on macOS). Run this script on a macOS host."
    exit 1
fi

build_app_for_arch() {
    local arch="$1"        # arm64 / x64
    local pub_dir="publish-mac-$arch"
    if [ ! -d "$pub_dir" ]; then
        return 1
    fi

    echo ""
    echo "🍎 Building $APP_NAME.app ($arch)..."

    local stage_dir
    stage_dir=$(mktemp -d -t soltts_dmg_XXXXXX)
    local app_root="$stage_dir/$APP_NAME.app"
    local contents="$app_root/Contents"
    local macos_dir="$contents/MacOS"
    local resources_dir="$contents/Resources"

    mkdir -p "$macos_dir" "$resources_dir"

    # 1. Copy the entire publish directory into Resources/app (including sol-server, sol-cli, resources, tools, config, etc.)
    mkdir -p "$resources_dir/app"
    # Use ditto to preserve permissions and symlinks
    ditto "$pub_dir/" "$resources_dir/app/"

    # Ensure executable permissions
    chmod +x "$resources_dir/app/sol-server" 2>/dev/null || true
    chmod +x "$resources_dir/app/sol-cli" 2>/dev/null || true
    chmod +x "$resources_dir/app/init-env-mac.sh" 2>/dev/null || true

    # 2. Launcher script: starts sol-server and opens the browser when the .app is double-clicked
    #    Important: the launcher must stay in the foreground (exec replaces it with the sol-server process),
    #    so macOS shows the Dock icon and allows closing via the usual means (Cmd+Q / right-click Dock > Quit).
    cat > "$macos_dir/$APP_NAME" <<'LAUNCHER_EOF'
#!/bin/bash
# SolTTS .app launcher
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
APP_RES="$DIR/../Resources/app"

cd "$APP_RES"

# Remove quarantine (harmless after the first run)
xattr -dr com.apple.quarantine "$APP_RES" 2>/dev/null || true

# Log directory
LOG_DIR="$HOME/Library/Logs/SolTTS"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sol-server.log"

# If a server with the same name is already running, just open the browser and exit (avoid port conflicts)
if pgrep -f "$APP_RES/sol-server" >/dev/null 2>&1; then
    echo "$(date) SolTTS is already running; opening the browser only" >> "$LOG_FILE"
    PORT="${SOLTTS_PORT:-5000}"
    open "http://localhost:$PORT/" || true
    exit 0
fi

# Wait asynchronously in the background for the port, then open the browser (without blocking the foreground process)
PORT="${SOLTTS_PORT:-5000}"
(
    for i in {1..60}; do
        if curl -s "http://localhost:$PORT/" >/dev/null 2>&1; then
            open "http://localhost:$PORT/" || true
            exit 0
        fi
        sleep 1
    done
) &

# Important: use exec to replace the current shell process with sol-server,
# so the .app's main process (Contents/MacOS/SolTTS) stays alive,
# the Dock shows an icon, and Cmd+Q / right-click Dock > Quit both close the process.
exec "$APP_RES/sol-server" >> "$LOG_FILE" 2>&1
LAUNCHER_EOF
    chmod +x "$macos_dir/$APP_NAME"

    # 3. Info.plist
    #    LSUIElement=false / LSBackgroundOnly=false explicitly declare this a regular GUI app,
    #    ensuring the process appears in the Dock and supports Cmd+Q to quit.
    cat > "$contents/Info.plist" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>           <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>    <string>$APP_NAME</string>
    <key>CFBundleExecutable</key>     <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>     <string>$BUNDLE_ID</string>
    <key>CFBundleVersion</key>        <string>${VERSION#v}</string>
    <key>CFBundleShortVersionString</key><string>${VERSION#v}</string>
    <key>CFBundlePackageType</key>    <string>APPL</string>
    <key>CFBundleSignature</key>      <string>????</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>LSMinimumSystemVersion</key> <string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSUIElement</key>            <false/>
    <key>LSBackgroundOnly</key>       <false/>
    <key>NSSupportsAutomaticTermination</key><false/>
    <key>NSSupportsSuddenTermination</key><false/>
</dict>
</plist>
PLIST_EOF

    # 4. App icon (if present)
    if [ -f "images/logo.png" ]; then
        if command -v sips &> /dev/null && command -v iconutil &> /dev/null; then
            local iconset="$stage_dir/AppIcon.iconset"
            mkdir -p "$iconset"
            for sz in 16 32 64 128 256 512; do
                sips -z $sz $sz images/logo.png --out "$iconset/icon_${sz}x${sz}.png" >/dev/null 2>&1 || true
                local d=$((sz * 2))
                sips -z $d $d images/logo.png --out "$iconset/icon_${sz}x${sz}@2x.png" >/dev/null 2>&1 || true
            done
            iconutil -c icns "$iconset" -o "$resources_dir/AppIcon.icns" 2>/dev/null || true
            if [ -f "$resources_dir/AppIcon.icns" ]; then
                /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string AppIcon" "$contents/Info.plist" 2>/dev/null || true
            fi
        fi
    fi

    # 5. Strip quarantine attributes
    xattr -dr com.apple.quarantine "$app_root" 2>/dev/null || true

    # 6. Create the DMG: use a temporary staging folder containing the .app and an Applications symlink
    local dmg_stage="$stage_dir/dmg"
    mkdir -p "$dmg_stage"
    ditto "$app_root" "$dmg_stage/$APP_NAME.app"
    ln -s /Applications "$dmg_stage/Applications"

    local dmg_path="$OUTPUT_DIR/SolTTS-$VERSION-macOS-$arch.dmg"
    rm -f "$dmg_path"

    echo "🛠  Creating DMG -> $dmg_path"
    hdiutil create \
        -volname "$APP_NAME $VERSION" \
        -srcfolder "$dmg_stage" \
        -ov -format UDZO \
        "$dmg_path" >/dev/null

    echo "✅ DMG created: $dmg_path"

    # Clean up temporary directory
    rm -rf "$stage_dir"
}

BUILT_ANY=0
if build_app_for_arch "arm64"; then BUILT_ANY=1; fi
if build_app_for_arch "x64"; then BUILT_ANY=1; fi

if [ $BUILT_ANY -eq 0 ]; then
    echo "❌ Neither publish-mac-arm64 nor publish-mac-x64 found. Run ./publish-mac.sh first."
    exit 1
fi

echo ""
echo "🎉 All DMG packages created! Output directory: $(cd "$OUTPUT_DIR" && pwd)"
