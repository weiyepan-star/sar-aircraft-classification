#!/usr/bin/env bash
# ============================================================================
#  编译 ACM 双栏技术报告（4 遍：pdflatex → bibtex → pdflatex → pdflatex）
#  用法： bash build.sh
# ============================================================================
set -u
cd "$(dirname "$0")"

# 清掉旧的中间文件，避免跨轮次残留导致的引用错乱
rm -f main.aux main.bbl main.blg main.log main.out main.lof main.lot main.toc

echo "===== 第 1 遍 pdflatex ====="
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null 2>&1
echo "exit=$?"

echo "===== bibtex ====="
bibtex main > /dev/null 2>&1
echo "exit=$?"

for p in 2 3 4; do
  echo "===== 第 $p 遍 pdflatex ====="
  pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
  echo "exit=$?"
done

echo ""
echo "===== 质量报告 ====="
if [ ! -f main.pdf ]; then
  echo "❌ 未生成 PDF，请看 main.log 末尾"
  tail -40 main.log
  exit 1
fi
printf '  页数            : %s\n' "$(pdfinfo main.pdf | awk '/^Pages/{print $2}')"
printf '  页面尺寸        : %s\n' "$(pdfinfo main.pdf | awk -F: '/^Page size/{print $2}' | sed 's/^ //')"
printf '  Overfull hbox   : %s\n' "$(grep -c 'Overfull \\hbox' main.log)"
printf '  Underfull hbox  : %s\n' "$(grep -c 'Underfull \\hbox' main.log)"
printf '  Overfull vbox   : %s\n' "$(grep -c 'Overfull \\vbox' main.log)"
printf '  Underfull vbox  : %s\n' "$(grep -c 'Underfull \\vbox' main.log)"
printf '  缺字            : %s\n' "$(grep -ci 'missing character' main.log)"
printf '  未定义引用/引文 : %s\n' "$(grep -c 'undefined' main.log)"
printf '  BibTeX 警告     : %s\n' "$(grep -c 'Warning--' main.blg 2>/dev/null || echo 0)"
printf '  字体嵌入        : %s\n' "$(pdffonts main.pdf | awk 'NR>2 && $0!="" {n++; if ($(NF-3)=="yes") e++} END {print e"/"n" 已嵌入"}')"
