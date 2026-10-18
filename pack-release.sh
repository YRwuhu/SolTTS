#!/bin/bash

# SolTTS resource and release packaging script (Linux & macOS)
# Automatically detects available publish-* directories and packages them

VERSION="v1.2.2"
RELEASE_DIR="releases"

mkdir -p "$RELEASE_DIR"

echo "📦 Packaging SolTTS $VERSION release files..."

# ----------- 1. Package the Linux bundle -----------
if [ -d "publish-linux" ]; then
    LINUX_TAR="$RELEASE_DIR/SolTTS-$VERSION-linux64.tar.gz"
    rm -f "$LINUX_TAR"
    echo -e "\nCompressing Linux bundle -> $LINUX_TAR ..."

    cp init-env.sh publish-linux/
    chmod +x publish-linux/init-env.sh

    (cd publish-linux && tar -czf "../$LINUX_TAR" ./*)

    echo "✅ Linux bundle (.tar.gz) packaging complete!"
else
    echo -e "\n⚠️  publish-linux directory not found; skipping Linux packaging. Hint: run ./publish-linux.sh first"
fi

# ----------- 2. Package the macOS bundle (arm64) -----------
pack_mac() {
    local arch="$1"          # arm64 / x64
    local arch_label="$2"    # arm64 / x64 (used in the file name)
    local pub_dir="publish-mac-$arch"
    if [ ! -d "$pub_dir" ]; then
        return
    fi

    local mac_tar="$RELEASE_DIR/SolTTS-$VERSION-macOS-$arch_label.tar.gz"
    rm -f "$mac_tar"
    echo -e "\nCompressing macOS ($arch_label) bundle -> $mac_tar ..."

    cp init-env-mac.sh "$pub_dir/"
    chmod +x "$pub_dir/init-env-mac.sh"

    (cd "$pub_dir" && tar -czf "../$mac_tar" ./*)

    echo "✅ macOS ($arch_label) bundle packaging complete!"
}

pack_mac "arm64" "arm64"
pack_mac "x64" "x64"

if [ ! -d "publish-mac-arm64" ] && [ ! -d "publish-mac-x64" ]; then
    echo -e "\n⚠️  publish-mac-* directory not found; skipping macOS packaging. Hint: run ./publish-mac.sh first"
fi

# ----------- 3. Build the macOS DMG (if publish-mac-* exists) -----------
if [ -f "pack-mac-dmg.sh" ]; then
    if [ -d "publish-mac-arm64" ] || [ -d "publish-mac-x64" ]; then
        echo -e "\n📦 Creating macOS .dmg installers..."
        chmod +x pack-mac-dmg.sh
        ./pack-mac-dmg.sh "$VERSION" "$RELEASE_DIR" || echo "⚠️  DMG packaging failed; check the output of pack-mac-dmg.sh."
    fi
fi

# ----------- 4. Resource bundle (resources-minimal) -----------
if [ -d "resources-minimal" ]; then
    RES_TAR="$RELEASE_DIR/SolTTS-resources-minimal-$VERSION.tar.gz"
    rm -f "$RES_TAR"
    echo -e "\nCompressing standalone resource bundle (resources-minimal) -> $RES_TAR ..."
    tar -czf "$RES_TAR" resources-minimal
    echo "✅ Standalone resource bundle packaging complete!"
fi

echo -e "\n🎉 All packaging jobs are in: $(cd "$RELEASE_DIR" && pwd)"
echo -e "Hint: you can now upload the files in '$RELEASE_DIR' to the Releases page."
