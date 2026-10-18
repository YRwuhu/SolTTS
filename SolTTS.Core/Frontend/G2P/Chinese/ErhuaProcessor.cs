using System;
using System.Collections.Generic;

namespace SolTTS.Core.Frontend.G2P.Chinese
{
    /// <summary>
    /// Erhua (r-coloring) processor.
    /// Ported from Genie-TTS Erhua.py.
    /// </summary>
    public class ErhuaProcessor
    {
        private static readonly HashSet<string> MustErhua = new HashSet<string>
        {
            "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"
        };

        private static readonly HashSet<string> NotErhua = new HashSet<string>
        {
            "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿",
            "妻儿", "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿",
            "脑瘫儿", "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿",
            "侄儿", "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿",
            "猪儿", "猫儿", "狗儿", "少儿",
        };

        public (List<string> initials, List<string> finals) MergeErhua(
            List<string> initials,
            List<string> finals,
            string word,
            string pos)
        {
            // 1. Fix er1 to er2 (when '儿' is word-final and pronounced er1)
            if (finals.Count > 0 && word.Length == finals.Count && word[^1] == '儿' && finals[^1] == "er1")
            {
                finals[^1] = "er2";
            }

            // 2. Check whether erhua processing should be skipped
            if (!MustErhua.Contains(word) && (NotErhua.Contains(word) || pos == "a" || pos == "j" || pos == "nr"))
            {
                return (initials, finals);
            }

            // 3. Length validation
            if (finals.Count != word.Length)
            {
                return (initials, finals);
            }

            // 4. Apply the erhua merge logic
            var newInitials = new List<string>();
            var newFinals = new List<string>();

            for (int i = 0; i < finals.Count; i++)
            {
                string phn = finals[i];
                // Whether to merge the erhua sound
                if (i == finals.Count - 1 &&
                    word[i] == '儿' &&
                    (phn == "er2" || phn == "er5") &&
                    word.Length >= 2 &&
                    !NotErhua.Contains(word[^2..]) &&
                    newFinals.Count > 0)
                {
                    // Give 'er' the tone of the preceding character
                    char tone = newFinals[^1][^1];
                    phn = "er" + (char.IsDigit(tone) ? tone : '5');
                }
                newInitials.Add(initials[i]);
                newFinals.Add(phn);
            }

            return (newInitials, newFinals);
        }
    }
}
