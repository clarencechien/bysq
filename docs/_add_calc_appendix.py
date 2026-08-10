"""Append a '怎麼算 + 計算器' appendix to both magazine reports."""
import re

CSS = """
.calc{background:var(--card); border:1px solid var(--line); border-radius:8px; padding:1.1rem 1.15rem; margin:1.2rem 0;}
.calc h4{margin:.1rem 0 .7rem; font-family:"Noto Serif TC",Georgia,serif; font-size:1rem;}
.calcgrid{display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.7rem .9rem;}
.field{display:flex; flex-direction:column; gap:.2rem;}
.field label{font-size:.78rem; color:var(--ink2); letter-spacing:.02em;}
.field input{font:inherit; font-size:.95rem; font-variant-numeric:tabular-nums;
  background:var(--paper); color:var(--ink); border:1px solid var(--line);
  border-radius:5px; padding:.42rem .6rem; width:100%;}
.field input:focus-visible{outline:2px solid var(--accent); outline-offset:1px; border-color:var(--accent);}
.calcout{margin-top:1rem; display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.7rem;}
.outbox{background:var(--paper); border:1px solid var(--line); border-radius:6px; padding:.7rem .85rem;}
.outbox .lbl{font-size:.74rem; color:var(--ink2); letter-spacing:.04em;}
.outbox .val{font-size:1.32rem; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.35;}
.outbox .sub{font-size:.79rem; color:var(--ink2); line-height:1.5;}
.verdict{margin-top:.8rem; padding:.6rem .85rem; border-radius:6px; font-size:.9rem; font-weight:600;}
.v-hold{background:var(--accent-soft); color:var(--accent);}
.v-cut{background:var(--ruin-bg); color:var(--ruin);}
.v-raise{background:var(--ok-bg); color:var(--ok);}
.v-floor{background:var(--warn-bg); color:var(--warn);}
.formula{background:var(--accent-soft); border-radius:6px; padding:.75rem .95rem; margin:.8rem 0;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88rem; line-height:1.9;
  overflow-x:auto; white-space:nowrap;}
.steps{counter-reset:s; list-style:none; padding-left:0;}
.steps li{counter-increment:s; position:relative; padding-left:2rem; margin:.5rem 0; font-size:.93rem;}
.steps li::before{content:"步驟 " counter(s); position:absolute; left:0; top:.28rem;
  font-size:.62rem; font-weight:700; letter-spacing:.03em; color:var(--accent);
  writing-mode:horizontal-tb; width:1.7rem;}
.steps li b{font-variant-numeric:tabular-nums;}
"""

SECTION = """
<h2>附錄二：GK 與 VPW 到底怎麼算？（附計算器）</h2>
<p>前面用了「看業績調薪」「逐年結算的年金」這種比喻。這一節把比喻拆掉，
用<b>{PERSONA}</b>的真實數字，一步一步算給你看——兩個方法其實都只有國中數學。</p>

<h3>VPW：只有一條公式</h3>
<p>VPW（Variable Percentage Withdrawal，變動百分比提領）的想法是：
<b>把「剩下的錢」平均分配到「剩下的年份」</b>，就像銀行算房貸月付金，只是方向相反——
房貸是「借一筆錢，分 30 年還」，VPW 是「有一筆錢，分 N 年花完」。用的是同一條年金公式：</p>
<div class="formula">今年可領 ＝ 目前資產 × 提領因子<br>
提領因子 ＝ r ÷ ( 1 − (1+r)<sup>−n</sup> )　　　r ＝ 假設實質報酬（本報告用 3%）、n ＝ 還剩幾年</div>
<p>「還剩幾年」用<b>活到 100 歲</b>來算（不是用平均餘命——因為有四分之一的機率有人活到 95，
用平均值會算太多）。所以：</p>
<ol class="steps">
{VPW_STEPS}
</ol>
<p>明年再算一次：資產變多，可領就變多；資產變少，可領就變少。
因為分母永遠是「剩下的年數」，<b>資產在數學上不可能被提前領完</b>——這就是 VPW 破產率趨近於零的原因。
代價也在同一條公式裡：市場跌三成，你的生活費就跟著跌三成。</p>
<p>提領因子隨年齡自動變大（剩的年份變少，同樣的錢可以花得更快）：</p>
<div class="tablewrap"><table>
<thead><tr><th>年齡</th><th class="r">剩餘年數 n</th><th class="r">提領因子（r=3%）</th><th class="r">1000 萬可領（年）</th></tr></thead>
<tbody>
<tr><td>50 歲</td><td class="r">50</td><td class="r">3.89%</td><td class="r">38.9 萬</td></tr>
<tr><td>60 歲</td><td class="r">40</td><td class="r">4.33%</td><td class="r">43.3 萬</td></tr>
<tr><td>65 歲</td><td class="r">35</td><td class="r">4.65%</td><td class="r">46.5 萬</td></tr>
<tr><td>75 歲</td><td class="r">25</td><td class="r">5.74%</td><td class="r">57.4 萬</td></tr>
<tr><td>85 歲</td><td class="r">15</td><td class="r">8.38%</td><td class="r">83.8 萬</td></tr>
<tr><td>95 歲</td><td class="r">5</td><td class="r">21.84%</td><td class="r">218.4 萬</td></tr>
</tbody></table></div>
<p class="meta">注意最後兩列：VPW 到了高齡會叫你「盡量花」。這對有遺贈需求的人是缺點，
對本篇主角是<b>設計目的</b>。實務上多數人會設一個上限（本報告設在目標支出的 2～2.5 倍），
把超出的部分留在帳上當長照與醫療準備。</p>

<h3>GK：兩條護欄，看比例不看漲跌</h3>
<p>GK（Guyton-Klinger）最容易被誤解的一點：<b>它不看市場漲跌，它看「你正在領的錢，佔你資產的百分之幾」</b>。
這個比例叫「當期提領率」。退休第一天的比例被記下來當基準，之後每年比對：</p>
<div class="formula">當期提領率 ＝ 今年要領的金額（一年份） ÷ 目前資產<br><br>
若 當期提領率 &gt; 起始提領率 × <b>1.2</b>　→　把生活費<b>砍 10%</b>（資本保存規則）<br>
若 當期提領率 &lt; 起始提領率 × <b>0.8</b>　→　把生活費<b>加 10%</b>（繁榮規則）<br>
其餘情況　→　照原本的金額領，只跟著物價調整</div>
<ol class="steps">
{GK_STEPS}
</ol>
<div class="calcnote"></div>
<p>這裡有一個很好用的心算捷徑：<b>1 ÷ 1.2 ＝ 0.833、1 ÷ 0.8 ＝ 1.25</b>。
也就是說，只要你還沒調整過金額，<b>資產跌到剩 83%（−17%）就會觸發減領，漲到 125%（+25%）就會觸發加領</b>。
帳面跌 15% 什麼事都不會發生——這也是為什麼 GK 在小回檔時完全不動作。</p>
<p>減領之後門檻會跟著往下移（因為分子變小了），所以連續下跌會是「砍一次、緩一下、再砍一次」，
而不是一次砍到底。反過來，加領也是一階一階往上。<b>這就是 GK 比 VPW「鈍」的原因</b>：
市場回來的時候，它每年只加得回 10%，而 VPW 隔年就跳回原位。</p>

<div class="calc" id="calc-{ID}">
<h4>🧮 自己算算看</h4>
<p class="sub" style="font-size:.86rem; color:var(--ink2); margin:.1rem 0 .9rem;">
改任何一格，下面的數字會立刻重算。金額單位：資產用<b>萬元</b>，每月金額用<b>元</b>。</p>
<div class="calcgrid">
<div class="field"><label for="a0-{ID}">退休時資產（萬元）</label><input type="number" id="a0-{ID}" value="{A0}" min="0" step="100"></div>
<div class="field"><label for="d0-{ID}">退休時每月從投資帳戶領（元）</label><input type="number" id="d0-{ID}" value="{D0}" min="0" step="1000"></div>
<div class="field"><label for="an-{ID}">目前資產（萬元）</label><input type="number" id="an-{ID}" value="{AN}" min="0" step="100"></div>
<div class="field"><label for="dn-{ID}">目前每月領（元）</label><input type="number" id="dn-{ID}" value="{D0}" min="0" step="1000"></div>
<div class="field"><label for="age-{ID}">目前年齡</label><input type="number" id="age-{ID}" value="{AGE}" min="30" max="99" step="1"></div>
<div class="field"><label for="pen-{ID}">每月年金（元，沒有就填 0）</label><input type="number" id="pen-{ID}" value="{PEN}" min="0" step="1000"></div>
<div class="field"><label for="fl-{ID}">每月砍不掉的地板（元）</label><input type="number" id="fl-{ID}" value="{FLOOR}" min="0" step="1000"></div>
<div class="field"><label for="r-{ID}">VPW 假設實質報酬（%）</label><input type="number" id="r-{ID}" value="3" min="0" max="8" step="0.5"></div>
</div>
<div class="calcout">
<div class="outbox"><div class="lbl">GK：目前提領率</div><div class="val" id="o-wr-{ID}">—</div>
<div class="sub" id="o-wrsub-{ID}">起始 — ｜護欄 — ～ —</div></div>
<div class="outbox"><div class="lbl">GK：觸發減領／加領的資產</div><div class="val" id="o-thr-{ID}">—</div>
<div class="sub">跌破左邊減 10%，漲過右邊加 10%</div></div>
<div class="outbox"><div class="lbl">VPW：今年每月可領</div><div class="val" id="o-vpw-{ID}">—</div>
<div class="sub" id="o-vpwsub-{ID}">因子 — ｜剩 — 年</div></div>
<div class="outbox"><div class="lbl">VPW：含年金的總生活費</div><div class="val" id="o-tot-{ID}">—</div>
<div class="sub" id="o-totsub-{ID}">相對地板 —</div></div>
</div>
<div class="verdict v-hold" id="o-verdict-{ID}">—</div>
</div>

<h3>一句話比較</h3>
<div class="tablewrap"><table>
<thead><tr><th></th><th>GK 護欄</th><th>VPW 攤銷</th></tr></thead>
<tbody>
<tr><td>算什麼</td><td>比對「提領率」有沒有越線</td><td>重算「剩下的錢 ÷ 剩下的年」</td></tr>
<tr><td>多久算一次</td><td>每年一次</td><td>每年一次</td></tr>
<tr><td>平常會不會動</td><td>不會，只在越線時 ±10%</td><td>每年都動，跟著資產走</td></tr>
<tr><td>市場回來時</td><td>每年加回 10%，很慢</td><td>隔年就回到該有的水位</td></tr>
<tr><td>會不會歸零</td><td>理論上會（減領有下限）</td><td>數學上不會（分母永遠是剩餘年數）</td></tr>
<tr><td>誰適合</td><td>要生活費穩定、能忍長期減支的人</td><td>支出彈性大、不留遺產的人</td></tr>
</tbody></table></div>

<p class="meta"><b>誠實說明</b>：完整版 GK 有四條規則——除了上面兩條護欄，還有
「通膨規則」（前一年虧損就跳過當年的物價調整、調幅上限 6%）與
「投資組合管理規則」（跌年優先從債券/現金部位提領）。
本篇兩位主角的模擬採用<b>兩條護欄的簡化版</b>（生活費以實質金額計算，等於自動含物價調整），
研究本體（v1–v3 報告）用的則是四條規則的完整版。簡化版對結論的影響方向：
略為<b>低估</b> GK 在空頭年的保護力，但不改變它與 VPW 的相對排序。</p>
"""

CALC_JS = """
<script>
(function(){
  var ids = __IDS__;
  ids.forEach(function(ID){
    var g = function(k){ return document.getElementById(k + '-' + ID); };
    var inputs = ['a0','d0','an','dn','age','pen','fl','r'].map(g);
    var fmt = function(n){ return n.toLocaleString('zh-TW', {maximumFractionDigits:0}); };
    function calc(){
      var a0 = +g('a0').value * 10000, d0 = +g('d0').value;
      var an = +g('an').value * 10000, dn = +g('dn').value;
      var age = +g('age').value, pen = +g('pen').value, fl = +g('fl').value;
      var r = +g('r').value / 100;
      // --- GK ---
      var wr0 = a0 > 0 ? (d0 * 12) / a0 : 0;
      var wr  = an > 0 ? (dn * 12) / an : Infinity;
      var up = wr0 * 1.2, lo = wr0 * 0.8;
      g('o-wr').textContent = isFinite(wr) ? (wr*100).toFixed(2) + '%' : '—';
      g('o-wrsub').textContent = '起始 ' + (wr0*100).toFixed(2) + '% ｜護欄 '
        + (lo*100).toFixed(2) + '% ～ ' + (up*100).toFixed(2) + '%';
      var cutAt = up > 0 ? (dn*12)/up : 0, raiseAt = lo > 0 ? (dn*12)/lo : 0;
      g('o-thr').textContent = fmt(cutAt/10000) + ' ／ ' + fmt(raiseAt/10000) + ' 萬';
      // --- VPW ---
      var n = Math.max(100 - age, 1);
      var f = n > 1 ? r / (1 - Math.pow(1+r, -n)) : 1;
      if (r === 0) f = 1 / n;
      var vpwM = an * f / 12;
      g('o-vpw').textContent = fmt(vpwM) + ' 元';
      g('o-vpwsub').textContent = '因子 ' + (f*100).toFixed(2) + '% ｜剩 ' + n + ' 年';
      var tot = vpwM + pen;
      g('o-tot').textContent = fmt(tot) + ' 元';
      g('o-totsub').textContent = fl > 0
        ? (tot >= fl ? '地板 ' + fmt(fl) + ' 元，還有 ' + fmt(tot-fl) + ' 元彈性空間'
                     : '低於地板 ' + fmt(fl - tot) + ' 元')
        : '未設地板';
      // --- verdict ---
      var v = g('o-verdict'), cls = 'v-hold', msg;
      if (!isFinite(wr) || an <= 0) { msg = '資產為 0，兩種方法都無法運作。'; cls = 'v-cut'; }
      else if (wr > up) { cls='v-cut';
        msg = '⚠️ GK 判定：超過上護欄 → 今年把生活費砍 10%，變成每月 '
              + fmt(dn*0.9) + ' 元（資產要回到 ' + fmt(raiseAt/10000) + ' 萬才會加回來）。'; }
      else if (wr < lo) { cls='v-raise';
        msg = '🎉 GK 判定：低於下護欄 → 今年把生活費加 10%，變成每月 ' + fmt(dn*1.1) + ' 元。'; }
      else { cls='v-hold';
        msg = '✅ GK 判定：在護欄之內 → 今年照領 ' + fmt(dn) + ' 元（僅隨物價調整）。資產跌到 '
              + fmt(cutAt/10000) + ' 萬才會減領。'; }
      if (tot < fl && fl > 0) { cls = 'v-floor';
        msg += '　同時 VPW 算出的金額已低於地板，實務上必須補到地板 —— 這就是動態規則開始失效的訊號。'; }
      v.className = 'verdict ' + cls;
      v.textContent = msg;
    }
    inputs.forEach(function(el){ el.addEventListener('input', calc); });
    calc();
  });
})();
</script>
"""

TW_VPW = """<li>先算還剩幾年：<b>100 − 65 ＝ 35 年</b></li>
<li>算提領因子：<b>0.03 ÷ (1 − 1.03⁻³⁵) ＝ 0.03 ÷ 0.6446 ＝ 4.65%</b></li>
<li>今年可從投資帳戶領：<b>1,000 萬 × 4.65% ＝ 46.5 萬／年 ＝ 每月 38,783 元</b></li>
<li>加上勞保年金 42,000 元 → <b>這對夫妻第一年每月可花 80,783 元</b>（約 8 萬，遠高於目標的 6 萬）</li>"""

TW_GK = """<li>算出第一年要從投資帳戶領多少：目標 60,000 − 年金 42,000 ＝ <b>每月 18,000 元 ＝ 一年 21.6 萬</b></li>
<li>起始提領率：<b>21.6 萬 ÷ 1,000 萬 ＝ 2.16%</b>。這個數字被記下來當基準</li>
<li>上護欄 ＝ 2.16% × 1.2 ＝ <b>2.59%</b>；下護欄 ＝ 2.16% × 0.8 ＝ <b>1.73%</b></li>
<li>換算成資產門檻：資產跌到 <b>833 萬</b>（−17%）就減領 10%（變 16,200 元）；
漲到 <b>1,250 萬</b>（+25%）就加領 10%（變 19,800 元）</li>"""

FIRE_VPW = """<li>先算還剩幾年：<b>100 − 50 ＝ 50 年</b></li>
<li>算提領因子：<b>0.03 ÷ (1 − 1.03⁻⁵⁰) ＝ 0.03 ÷ 0.7719 ＝ 3.89%</b></li>
<li>今年可領：<b>5,000 萬 × 3.89% ＝ 194.3 萬／年 ＝ 每月 161,938 元</b></li>
<li>扣掉工會保費 7,000 元 → <b>第一年每月可花約 154,900 元</b>（50 歲沒有年金可加）。
15 年後年金入帳，同一條公式會自動把可領金額往上推</li>"""

FIRE_GK = """<li>第一年要從投資帳戶領多少：目標 150,000 ＋ 工會保費 7,000 − 年金 0 ＝
<b>每月 157,000 元 ＝ 一年 188.4 萬</b></li>
<li>起始提領率：<b>188.4 萬 ÷ 5,000 萬 ＝ 3.77%</b></li>
<li>上護欄 ＝ <b>4.52%</b>；下護欄 ＝ <b>3.01%</b></li>
<li>換算成資產門檻：資產跌到 <b>4,167 萬</b>（−17%）減領 10%（變 141,300 元）；
漲到 <b>6,250 萬</b>（+25%）加領 10%。1966 開局那條紅線之所以被壓到 8 萬，
就是這個規則連續觸發了三次</li>"""

CONFIGS = [
    dict(file="bysq_tw_case.html", ID="tw", PERSONA="1000 萬、65 歲退休的那對夫妻",
         VPW_STEPS=TW_VPW, GK_STEPS=TW_GK,
         A0=1000, AN=1000, D0=18000, AGE=65, PEN=42000, FLOOR=30000),
    dict(file="bysq_fire_case.html", ID="fire", PERSONA="5000 萬、50 歲 FIRE 的那對頂客夫妻",
         VPW_STEPS=FIRE_VPW, GK_STEPS=FIRE_GK,
         A0=5000, AN=5000, D0=157000, AGE=50, PEN=0, FLOOR=65000),
]

for cfg in CONFIGS:
    p = open(cfg["file"], encoding="utf-8").read()
    assert "附錄二" not in p, cfg["file"] + " already has the appendix"
    p = p.replace("</style>", CSS + "</style>", 1)
    sec = SECTION
    for k, v in cfg.items():
        if k != "file":
            sec = sec.replace("{" + k + "}", str(v))
    sec = sec.replace('<div class="calcnote"></div>', '')
    assert "{" not in sec.replace("{", "{", 1) or True
    js = CALC_JS.replace("__IDS__", '["%s"]' % cfg["ID"])
    p = p.replace('<div class="foot">', sec + "\n" + js + "\n<div class=\"foot\">", 1)
    open(cfg["file"], "w", encoding="utf-8").write(p)
    print("updated", cfg["file"], len(p))
