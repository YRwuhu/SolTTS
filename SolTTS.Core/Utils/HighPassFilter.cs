using System;

namespace SolTTS.Core.Utils
{
    /// <summary>
    /// Simple first-order high-pass filter that removes DC offset
    /// </summary>
    public class HighPassFilter
    {
        private float _prevInput;
        private float _prevOutput;
        private readonly float _alpha;

        public HighPassFilter(float sampleRate, float cutoffHz = 20f)
        {
            float rc = 1f / (2f * MathF.PI * cutoffHz);
            float dt = 1f / sampleRate;
            _alpha = rc / (rc + dt);
        }

        public void Process(Span<float> samples)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                float input = samples[i];
                float output = _alpha * (_prevOutput + input - _prevInput);
                _prevInput = input;
                _prevOutput = output;
                samples[i] = output;
            }
        }
    }
}
