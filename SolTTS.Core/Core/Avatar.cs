namespace SolTTS.Core.Core
{
    /// <summary>
    /// Represents an available voice/character configuration.
    /// </summary>
    public class Avatar
    {
        /// <summary>
        /// Unique voice identifier.
        /// </summary>
        public string Id { get; set; } = "";

        /// <summary>
        /// Voice display name.
        /// </summary>
        public string Name { get; set; } = "";

        /// <summary>
        /// Voice description.
        /// </summary>
        public string? Description { get; set; }

        /// <summary>
        /// Engine version used by this voice. true = V2, false = V1.
        /// Falls back to the global TTSConfig.UseEngineV2 when not specified.
        /// </summary>
        public bool? UseEngineV2 { get; set; }

        /// <summary>
        /// Default/fallback reference audio ID. Used when the request does not specify one.
        /// </summary>
        public string DefaultReferenceId { get; set; } = "default";

        /// <summary>
        /// All reference audio available for this voice.
        /// </summary>
        public List<ReferenceAudio> References { get; set; } = new();

        /// <summary>
        /// Get the default reference audio.
        /// </summary>
        public ReferenceAudio? GetDefaultReference()
        {
            return References.FirstOrDefault(r => r.Id == DefaultReferenceId)
                   ?? References.FirstOrDefault();
        }

        /// <summary>
        /// Get the reference audio with the given ID; falls back to the default when not found.
        /// </summary>
        public ReferenceAudio? GetReference(string? referenceId)
        {
            if (string.IsNullOrEmpty(referenceId))
                return GetDefaultReference();

            return References.FirstOrDefault(r => r.Id == referenceId)
                   ?? GetDefaultReference();
        }
    }

    /// <summary>
    /// Represents a reference audio configuration.
    /// </summary>
    public class ReferenceAudio
    {
        /// <summary>
        /// Unique reference audio identifier (unique within the Avatar).
        /// </summary>
        public string Id { get; set; } = "default";

        /// <summary>
        /// Reference audio display name.
        /// </summary>
        public string? Name { get; set; }

        /// <summary>
        /// Reference audio path. Relative paths are resolved against the Avatar's references directory.
        /// Example: "normal.wav" -> "{AvatarsDir}/{avatarId}/references/normal.wav"
        /// </summary>
        public string AudioPath { get; set; } = "";

        /// <summary>
        /// Text content corresponding to the reference audio.
        /// </summary>
        public string Text { get; set; } = "";

        /// <summary>
        /// Language tag (optional).
        /// </summary>
        public string? Language { get; set; }

        /// <summary>
        /// Get the full audio path. Relative paths are joined with the Avatar's references directory.
        /// </summary>
        /// <param name="avatarsDir">Voice directory root</param>
        /// <param name="avatarId">Voice ID</param>
        public string GetFullAudioPath(string avatarsDir, string avatarId)
        {
            if (string.IsNullOrEmpty(AudioPath))
                return "";

            // 1. Return absolute paths directly
            if (Path.IsPathRooted(AudioPath))
                return AudioPath;

            // 2. Check whether it is already a reachable path relative to the current directory
            if (File.Exists(AudioPath))
                return Path.GetFullPath(AudioPath);

            // 3. Default behavior: join {avatarsDir}/{avatarId}/references/{AudioPath}
            return Path.Combine(avatarsDir, avatarId, "references", AudioPath);
        }
    }
}
