# -*- coding: utf-8 -*-
"""经原大论文表3-1至表3-6及data.xlsx字段说明核实的临床单位。"""

UNITS = {
    "年龄": "岁",
    "BMI": r"kg/m$^2$",
    "心率": "次/分",
    "舒张压": "mmHg",
    "收缩压": "mmHg",
    "肌酸激酶": "U/L",
    "乳酸脱氢酶": "U/L",
    "肌酸激酶同工酶": "U/L",
    "血清肌钙蛋白T": "ng/mL",
    # 下列三项为根据血细胞计数公式推导的量纲，而非病历直接记录单位。
    "SII": r"$\times10^9$/L",
    "SIRI": r"$\times10^9$/L",
    "AISI": r"$\times10^{18}$/L$^2$",
    "丙氨酸氨基转移酶": "U/L",
    "天门冬氨酸氨基转移酶": "U/L",
    "谷氨酰转肽酶": "U/L",
    "总胆红素": r"$\mu$mol/L",
    "白蛋白": "g/L",
    "肌酐": r"$\mu$mol/L",
    "尿素": "mmol/L",
    "尿酸": r"$\mu$mol/L",
    "总胆固醇": "mmol/L",
    "甘油三酯": "mmol/L",
    "低密度脂蛋白": "mmol/L",
    "高密度脂蛋白": "mmol/L",
    "TC-HDLDL": "mmol/L",
    "血钾": "mmol/L",
    "血氯": "mmol/L",
    "血钠": "mmol/L",
    "血钙": "mmol/L",
    "葡萄糖": "mmol/L",
    "PR间期": "ms",
    "QRS时限": "ms",
    "QTc间期": "ms",
    "T Axis": r"$^\circ$",
    "左室射血分数": r"\%",
    "心指数": r"L/min/m$^2$",
    "左室舒张末内径": "mm",
    "心室收缩容量": "mL",
    "室壁运动评分": "分",
}


# 保留原始数据库字段名以便复现，但在表格和图中使用临床标准名称。
DISPLAY_NAMES = {
    "心室收缩容量": "左心室收缩末期容积（LVESV）",
    "低密度脂蛋白": "低密度脂蛋白胆固醇",
}


def display_name(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)


def format_label_unit(label: str, unit: str | None) -> str:
    if not unit:
        return label
    if label.endswith("）") and "（" in label:
        return f"{label[:-1]}，{unit}）"
    return f"{label}（{unit}）"


def append_unit(label: str) -> str:
    unit = UNITS.get(label)
    shown = display_name(label)
    return format_label_unit(shown, unit)
