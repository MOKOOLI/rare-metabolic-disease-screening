# آپلود روی گیت‌هاب (مخزن MOKOOLI/rare-metabolic-disease-screening)

فایل zip را باز کن و داخل پوشه‌اش ترمینال بزن:

```bash
git init
git add .
git commit -m "Rare metabolic disease screening pipeline v1.0"
git branch -M main
git remote add origin https://github.com/MOKOOLI/rare-metabolic-disease-screening.git
git push -u origin main
```

اگر موقع ساخت مخزن README یا LICENSE ساخته بودی و push خطا داد:
```bash
git pull origin main --allow-unrelated-histories --no-edit
git push -u origin main
```

گیت‌هاب به‌جای رمز عبور، **Personal Access Token** می‌خواهد: Settings ← Developer settings ← Tokens.

**نکته‌ی مهم:** دستور `curl` راهنمای قبلی‌ات فایل COPYING خود Git را دانلود می‌کرد که مجوز **GPL** است، نه MIT. فایل LICENSE داخل این پروژه MIT درست است.

بعد از push:
1. در صفحه‌ی مخزن، روی چرخ‌دنده‌ی About کلیک کن. Description را وارد کن و این Topicها را اضافه کن: `metabolomics` `mass-spectrometry` `newborn-screening` `machine-learning` `clinical-biochemistry`
2. اسم و ORCID خودت را در `CITATION.cff` بنویس.
3. اپ را رایگان روی share.streamlit.io دیپلوی کن و لینکش را در README بگذار.
4. مخزن را در پروفایل گیت‌هاب Pin کن.
