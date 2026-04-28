from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


SEED = 20260427
TOTAL = 1500
TRAIN_N = 1050
DEV_N = 225
TEST_N = 225


@dataclass(frozen=True)
class Term:
    zh: str
    en: str
    label: str
    aliases: tuple[str, ...] = ()


TERMS: list[Term] = [
    # Herbs
    Term("黄芪", "Astragalus root", "HERB", ("Astragali Radix",)),
    Term("桂枝", "Cinnamon twig", "HERB", ("Cinnamomi Ramulus",)),
    Term("白芍", "White peony root", "HERB", ("Paeoniae Radix Alba",)),
    Term("炙甘草", "Honey-fried licorice root", "HERB", ("Glycyrrhizae Radix Praeparata",)),
    Term("生姜", "Fresh ginger", "HERB", ("Zingiberis Rhizoma Recens",)),
    Term("大枣", "Jujube", "HERB", ("Ziziphi Fructus",)),
    Term("人参", "Ginseng", "HERB", ("Ginseng Radix",)),
    Term("党参", "Codonopsis root", "HERB", ("Codonopsis Radix",)),
    Term("当归", "Chinese angelica root", "HERB", ("Angelicae Sinensis Radix",)),
    Term("川芎", "Chuanxiong rhizome", "HERB", ("Chuanxiong Rhizoma",)),
    Term("熟地黄", "Prepared rehmannia root", "HERB", ("Rehmanniae Radix Praeparata",)),
    Term("生地黄", "Raw rehmannia root", "HERB", ("Rehmanniae Radix",)),
    Term("茯苓", "Poria", "HERB", ("Poria cocos",)),
    Term("白术", "Atractylodes macrocephala rhizome", "HERB", ("Atractylodis Macrocephalae Rhizoma",)),
    Term("泽泻", "Alisma rhizome", "HERB", ("Alismatis Rhizoma",)),
    Term("牡丹皮", "Moutan bark", "HERB", ("Moutan Cortex",)),
    Term("半夏", "Pinellia tuber", "HERB", ("Pinelliae Rhizoma",)),
    Term("陈皮", "Tangerine peel", "HERB", ("Citri Reticulatae Pericarpium",)),
    Term("麦冬", "Ophiopogon root", "HERB", ("Ophiopogonis Radix",)),
    Term("五味子", "Schisandra fruit", "HERB", ("Schisandrae Fructus",)),
    Term("柴胡", "Bupleurum root", "HERB", ("Bupleuri Radix",)),
    Term("黄芩", "Scutellaria root", "HERB", ("Scutellariae Radix",)),
    Term("栀子", "Gardenia fruit", "HERB", ("Gardeniae Fructus",)),
    Term("连翘", "Forsythia fruit", "HERB", ("Forsythiae Fructus",)),
    Term("金银花", "Honeysuckle flower", "HERB", ("Lonicerae Japonicae Flos",)),
    Term("桔梗", "Platycodon root", "HERB", ("Platycodonis Radix",)),
    Term("杏仁", "Apricot kernel", "HERB", ("Armeniacae Semen Amarum",)),
    Term("麻黄", "Ephedra", "HERB", ("Ephedrae Herba",)),
    Term("石膏", "Gypsum", "HERB", ("Gypsum Fibrosum",)),
    Term("知母", "Anemarrhena rhizome", "HERB", ("Anemarrhenae Rhizoma",)),
    Term("黄连", "Coptis rhizome", "HERB", ("Coptidis Rhizoma",)),
    Term("黄柏", "Phellodendron bark", "HERB", ("Phellodendri Cortex",)),
    Term("龙胆草", "Chinese gentian root", "HERB", ("Gentianae Radix et Rhizoma",)),
    Term("丹参", "Salvia miltiorrhiza root", "HERB", ("Salviae Miltiorrhizae Radix",)),
    Term("桃仁", "Peach kernel", "HERB", ("Persicae Semen",)),
    Term("红花", "Safflower", "HERB", ("Carthami Flos",)),
    Term("牛膝", "Achyranthes root", "HERB", ("Achyranthis Bidentatae Radix",)),
    Term("杜仲", "Eucommia bark", "HERB", ("Eucommiae Cortex",)),
    Term("肉桂", "Cinnamon bark", "HERB", ("Cinnamomi Cortex",)),
    Term("附子", "Prepared aconite root", "HERB", ("Aconiti Lateralis Radix Praeparata",)),
    Term("干姜", "Dried ginger", "HERB", ("Zingiberis Rhizoma",)),
    Term("山药", "Chinese yam", "HERB", ("Dioscoreae Rhizoma",)),
    Term("山茱萸", "Cornus fruit", "HERB", ("Corni Fructus",)),
    Term("枸杞子", "Goji berry", "HERB", ("Lycii Fructus",)),
    Term("菊花", "Chrysanthemum flower", "HERB", ("Chrysanthemi Flos",)),
    Term("薄荷", "Mint", "HERB", ("Menthae Haplocalycis Herba",)),
    Term("防风", "Saposhnikovia root", "HERB", ("Saposhnikoviae Radix",)),
    Term("荆芥", "Schizonepeta", "HERB", ("Schizonepetae Herba",)),
    Term("羌活", "Notopterygium root", "HERB", ("Notopterygii Rhizoma et Radix",)),
    Term("独活", "Angelica pubescens root", "HERB", ("Angelicae Pubescentis Radix",)),
    Term("车前子", "Plantain seed", "HERB", ("Plantaginis Semen",)),
    Term("酸枣仁", "Sour jujube seed", "HERB", ("Ziziphi Spinosae Semen",)),
    Term("远志", "Polygala root", "HERB", ("Polygalae Radix",)),
    Term("天麻", "Gastrodia rhizome", "HERB", ("Gastrodiae Rhizoma",)),
    Term("钩藤", "Uncaria hook", "HERB", ("Uncariae Ramulus cum Uncis",)),
    Term("砂仁", "Amomum fruit", "HERB", ("Amomi Fructus",)),
    Term("木香", "Aucklandia root", "HERB", ("Aucklandiae Radix",)),
    Term("香附", "Cyperus rhizome", "HERB", ("Cyperi Rhizoma",)),
    Term("苍术", "Atractylodes rhizome", "HERB", ("Atractylodis Rhizoma",)),
    Term("厚朴", "Magnolia bark", "HERB", ("Magnoliae Officinalis Cortex",)),
    Term("薏苡仁", "Coix seed", "HERB", ("Coicis Semen",)),
    Term("蒲公英", "Dandelion", "HERB", ("Taraxaci Herba",)),
    # Formulas
    Term("黄芪桂枝五物汤", "Huangqi Guizhi Wuwu Decoction", "FORMULA"),
    Term("四君子汤", "Four-Gentlemen Decoction", "FORMULA"),
    Term("四物汤", "Four-Substance Decoction", "FORMULA"),
    Term("六味地黄丸", "Liuwei Dihuang Pill", "FORMULA"),
    Term("桂枝汤", "Cinnamon Twig Decoction", "FORMULA"),
    Term("麻黄汤", "Ephedra Decoction", "FORMULA"),
    Term("小青龙汤", "Minor Blue-Green Dragon Decoction", "FORMULA"),
    Term("白虎汤", "White Tiger Decoction", "FORMULA"),
    Term("小柴胡汤", "Minor Bupleurum Decoction", "FORMULA"),
    Term("补中益气汤", "Buzhong Yiqi Decoction", "FORMULA", ("Tonify the Middle and Augment Qi Decoction",)),
    Term("逍遥散", "Xiaoyao Powder", "FORMULA", ("Free Wanderer Powder",)),
    Term("归脾汤", "Guipi Decoction", "FORMULA", ("Restore the Spleen Decoction",)),
    Term("天王补心丹", "Tianwang Buxin Pill", "FORMULA", ("Heavenly Emperor Heart-Supplementing Elixir",)),
    Term("温胆汤", "Wendan Decoction", "FORMULA", ("Warm the Gallbladder Decoction",)),
    Term("二陈汤", "Erchen Decoction", "FORMULA", ("Two-Aged Decoction",)),
    Term("血府逐瘀汤", "Xuefu Zhuyu Decoction", "FORMULA", ("Drive Out Stasis in the Mansion of Blood Decoction",)),
    Term("银翘散", "Yinqiao Powder", "FORMULA", ("Honeysuckle and Forsythia Powder",)),
    Term("龙胆泻肝汤", "Longdan Xiegan Decoction", "FORMULA", ("Gentian Decoction to Drain the Liver",)),
    Term("生脉散", "Shengmai Powder", "FORMULA", ("Generate the Pulse Powder",)),
    Term("参苓白术散", "Shenling Baizhu Powder", "FORMULA"),
    Term("理中丸", "Lizhong Pill", "FORMULA"),
    Term("真武汤", "Zhenwu Decoction", "FORMULA"),
    Term("半夏泻心汤", "Banxia Xiexin Decoction", "FORMULA"),
    Term("苓桂术甘汤", "Linggui Zhugan Decoction", "FORMULA"),
    # Syndromes
    Term("气虚证", "qi deficiency pattern", "SYNDROME"),
    Term("血虚证", "blood deficiency pattern", "SYNDROME"),
    Term("阳虚证", "yang deficiency pattern", "SYNDROME"),
    Term("阴虚证", "yin deficiency pattern", "SYNDROME"),
    Term("气滞证", "qi stagnation pattern", "SYNDROME"),
    Term("血瘀证", "blood stasis pattern", "SYNDROME"),
    Term("痰湿证", "phlegm-dampness pattern", "SYNDROME"),
    Term("湿热证", "damp-heat pattern", "SYNDROME"),
    Term("风寒束表证", "wind-cold fettering the exterior pattern", "SYNDROME"),
    Term("风热犯表证", "wind-heat invading the exterior pattern", "SYNDROME"),
    Term("肝郁脾虚证", "liver depression and spleen deficiency pattern", "SYNDROME"),
    Term("脾胃虚寒证", "spleen-stomach deficiency cold pattern", "SYNDROME"),
    Term("心脾两虚证", "heart-spleen dual deficiency pattern", "SYNDROME"),
    Term("肾阴虚证", "kidney yin deficiency pattern", "SYNDROME"),
    Term("肾阳虚证", "kidney yang deficiency pattern", "SYNDROME"),
    Term("肺气虚证", "lung qi deficiency pattern", "SYNDROME"),
    Term("胃阴不足证", "stomach yin insufficiency pattern", "SYNDROME"),
    Term("肝火上炎证", "liver fire flaming upward pattern", "SYNDROME"),
    Term("寒湿困脾证", "cold-dampness encumbering the spleen pattern", "SYNDROME"),
    Term("气虚血瘀证", "qi deficiency with blood stasis pattern", "SYNDROME"),
    Term("气阴两虚证", "dual deficiency of qi and yin pattern", "SYNDROME"),
    Term("痰热内扰证", "internal disturbance of phlegm-heat pattern", "SYNDROME"),
    # Therapies
    Term("益气", "supplement qi", "THERAPY"),
    Term("养血", "nourish blood", "THERAPY"),
    Term("活血化瘀", "invigorate blood and resolve stasis", "THERAPY"),
    Term("疏肝解郁", "soothe the liver and relieve constraint", "THERAPY"),
    Term("健脾益气", "strengthen the spleen and supplement qi", "THERAPY"),
    Term("清热解毒", "clear heat and resolve toxin", "THERAPY"),
    Term("温阳散寒", "warm yang and disperse cold", "THERAPY"),
    Term("滋阴降火", "nourish yin and drain fire", "THERAPY"),
    Term("化痰止咳", "transform phlegm and relieve cough", "THERAPY"),
    Term("利水渗湿", "promote urination and leach out dampness", "THERAPY"),
    Term("和解少阳", "harmonize shaoyang", "THERAPY"),
    Term("解表散寒", "release the exterior and disperse cold", "THERAPY"),
    Term("清泻肝火", "clear and drain liver fire", "THERAPY"),
    Term("补肾填精", "tonify the kidney and replenish essence", "THERAPY"),
    Term("宁心安神", "calm the heart and quiet the spirit", "THERAPY"),
    Term("理气止痛", "regulate qi and relieve pain", "THERAPY"),
    Term("燥湿化痰", "dry dampness and transform phlegm", "THERAPY"),
    Term("养阴生津", "nourish yin and generate fluids", "THERAPY"),
    # Organs
    Term("肝", "liver", "ORGAN"),
    Term("心", "heart", "ORGAN"),
    Term("脾", "spleen", "ORGAN"),
    Term("肺", "lung", "ORGAN"),
    Term("肾", "kidney", "ORGAN"),
    Term("胃", "stomach", "ORGAN"),
    Term("胆", "gallbladder", "ORGAN"),
    Term("大肠", "large intestine", "ORGAN"),
    Term("小肠", "small intestine", "ORGAN"),
    # Symptoms
    Term("神疲乏力", "fatigue and lack of strength", "SYMPTOM"),
    Term("少气懒言", "shortness of breath and reluctance to speak", "SYMPTOM"),
    Term("面色萎黄", "sallow complexion", "SYMPTOM"),
    Term("心悸失眠", "palpitations and insomnia", "SYMPTOM"),
    Term("头晕目眩", "dizziness and blurred vision", "SYMPTOM"),
    Term("口干咽燥", "dry mouth and throat", "SYMPTOM"),
    Term("畏寒肢冷", "aversion to cold and cold limbs", "SYMPTOM"),
    Term("胸胁胀痛", "distending pain in the chest and hypochondrium", "SYMPTOM"),
    Term("咳嗽痰多", "cough with profuse phlegm", "SYMPTOM"),
    Term("纳差便溏", "poor appetite and loose stools", "SYMPTOM"),
    Term("腰膝酸软", "soreness and weakness of the lower back and knees", "SYMPTOM"),
    Term("舌红少苔", "red tongue with scant coating", "SYMPTOM"),
    Term("舌淡苔白", "pale tongue with white coating", "SYMPTOM"),
    Term("脉细弱", "fine and weak pulse", "SYMPTOM"),
    Term("脉弦数", "wiry and rapid pulse", "SYMPTOM"),
    Term("口苦咽干", "bitter taste and dry throat", "SYMPTOM"),
    Term("小便短赤", "short and reddish urination", "SYMPTOM"),
    Term("大便秘结", "constipation", "SYMPTOM"),
    # Diseases / topics
    Term("慢性胃炎", "chronic gastritis", "DISEASE"),
    Term("失眠", "insomnia", "DISEASE"),
    Term("眩晕", "vertigo", "DISEASE"),
    Term("咳嗽", "cough", "DISEASE"),
    Term("泄泻", "diarrhea", "DISEASE"),
    Term("胸痹", "chest impediment", "DISEASE"),
    Term("消渴", "wasting-thirst disorder", "DISEASE"),
    Term("月经不调", "irregular menstruation", "DISEASE"),
    Term("感冒", "common cold", "DISEASE"),
    Term("痹证", "impediment disease", "DISEASE"),
]


def by_label(label: str) -> list[Term]:
    return [t for t in TERMS if t.label == label]


HERBS = by_label("HERB")
FORMULAS = by_label("FORMULA")
SYNDROMES = by_label("SYNDROME")
THERAPIES = by_label("THERAPY")
ORGANS = by_label("ORGAN")
SYMPTOMS = by_label("SYMPTOM")
DISEASES = by_label("DISEASE")


def ent(term: Term) -> dict[str, str]:
    return {"zh": term.zh, "en": term.en, "type": term.label}


def make_templates(rng: random.Random):
    def t1():
        f, s = rng.choice(FORMULAS), rng.choice(SYNDROMES)
        a, b = rng.sample(SYMPTOMS, 2)
        return (
            f"{f.zh}常用于{s.zh}，症见{a.zh}、{b.zh}。",
            f"{f.en} is commonly used for {s.en}, with manifestations such as {a.en} and {b.en}.",
            "formula_indication",
            [ent(f), ent(s), ent(a), ent(b)],
        )

    def t2():
        h1, h2 = rng.sample(HERBS, 2)
        th, o = rng.choice(THERAPIES), rng.choice(ORGANS)
        return (
            f"{h1.zh}与{h2.zh}相伍，旨在{th.zh}，并调和{o.zh}功能。",
            f"{h1.en} and {h2.en} are combined to {th.en} and regulate {o.en} function.",
            "herb_pair",
            [ent(h1), ent(h2), ent(th), ent(o)],
        )

    def t3():
        d, s, th, f = rng.choice(DISEASES), rng.choice(SYNDROMES), rng.choice(THERAPIES), rng.choice(FORMULAS)
        return (
            f"对于辨证属{s.zh}的{d.zh}，治疗应以{th.zh}为主，可参考{f.zh}加减。",
            f"For {d.en} differentiated as {s.en}, treatment should mainly {th.en}, and {f.en} may be modified accordingly.",
            "disease_syndrome_treatment",
            [ent(d), ent(s), ent(th), ent(f)],
        )

    def t4():
        s, a, b, h = rng.choice(SYNDROMES), rng.choice(SYMPTOMS), rng.choice(SYMPTOMS), rng.choice(HERBS)
        th = rng.choice(THERAPIES)
        return (
            f"若{s.zh}兼见{a.zh}和{b.zh}，可酌加{h.zh}以{th.zh}。",
            f"When {s.en} is accompanied by {a.en} and {b.en}, {h.en} may be added to {th.en}.",
            "modification_rule",
            [ent(s), ent(a), ent(b), ent(h), ent(th)],
        )

    def t5():
        f, h1, h2 = rng.choice(FORMULAS), rng.choice(HERBS), rng.choice(HERBS)
        s = rng.choice(SYNDROMES)
        return (
            f"{f.zh}中{h1.zh}偏于扶正，{h2.zh}偏于祛邪，适合{s.zh}的复合病机。",
            f"In {f.en}, {h1.en} mainly supports healthy qi, while {h2.en} mainly dispels pathogenic factors, making the formula suitable for the complex pathogenesis of {s.en}.",
            "formula_composition",
            [ent(f), ent(h1), ent(h2), ent(s)],
        )

    def t6():
        o, s, th = rng.choice(ORGANS), rng.choice(SYNDROMES), rng.choice(THERAPIES)
        a, b = rng.sample(SYMPTOMS, 2)
        return (
            f"{o.zh}失调可表现为{s.zh}，临床常见{a.zh}、{b.zh}，治宜{th.zh}。",
            f"Dysfunction of the {o.en} may present as {s.en}; clinically, {a.en} and {b.en} are often observed, and treatment should {th.en}.",
            "organ_pattern",
            [ent(o), ent(s), ent(a), ent(b), ent(th)],
        )

    def t7():
        h, s, d = rng.choice(HERBS), rng.choice(SYNDROMES), rng.choice(DISEASES)
        return (
            f"{h.zh}在{d.zh}相关文本中常与{s.zh}同时出现，属于术语一致性评估的重点实体。",
            f"In texts related to {d.en}, {h.en} often co-occurs with {s.en}, making it a key entity for terminology consistency evaluation.",
            "translation_evaluation",
            [ent(h), ent(d), ent(s)],
        )

    def t8():
        f, th1, th2, s = rng.choice(FORMULAS), rng.choice(THERAPIES), rng.choice(THERAPIES), rng.choice(SYNDROMES)
        return (
            f"{f.zh}体现了{th1.zh}与{th2.zh}并用的思路，适用于{s.zh}。",
            f"{f.en} reflects the combined strategy of {th1.en} and {th2.en}, and is applicable to {s.en}.",
            "therapy_principle",
            [ent(f), ent(th1), ent(th2), ent(s)],
        )

    def t9():
        s, h1, h2, h3 = rng.choice(SYNDROMES), *rng.sample(HERBS, 3)
        return (
            f"针对{s.zh}，方中常配伍{h1.zh}、{h2.zh}和{h3.zh}以增强整体疗效。",
            f"For {s.en}, {h1.en}, {h2.en}, and {h3.en} are often combined in a formula to enhance the overall therapeutic effect.",
            "multi_herb_sentence",
            [ent(s), ent(h1), ent(h2), ent(h3)],
        )

    def t10():
        d, a, b, s = rng.choice(DISEASES), rng.choice(SYMPTOMS), rng.choice(SYMPTOMS), rng.choice(SYNDROMES)
        return (
            f"{d.zh}患者若出现{a.zh}、{b.zh}，多提示{s.zh}。",
            f"When patients with {d.en} present with {a.en} and {b.en}, it often indicates {s.en}.",
            "diagnostic_clue",
            [ent(d), ent(a), ent(b), ent(s)],
        )

    def t11():
        h, th, f = rng.choice(HERBS), rng.choice(THERAPIES), rng.choice(FORMULAS)
        return (
            f"在翻译{f.zh}相关句子时，{h.zh}的英文应保持稳定，以免影响{th.zh}语义。",
            f"When translating sentences related to {f.en}, the English rendering of {h.en} should remain stable to avoid altering the meaning of {th.en}.",
            "terminology_consistency",
            [ent(f), ent(h), ent(th)],
        )

    def t12():
        s, th, f, o = rng.choice(SYNDROMES), rng.choice(THERAPIES), rng.choice(FORMULAS), rng.choice(ORGANS)
        return (
            f"{s.zh}的治疗原则包括{th.zh}，并需兼顾{o.zh}的生理特点，代表方可选{f.zh}。",
            f"The treatment principle for {s.en} includes {th.en}; the physiological characteristics of the {o.en} should also be considered, and {f.en} may be selected as a representative formula.",
            "treatment_principle",
            [ent(s), ent(th), ent(o), ent(f)],
        )

    def t13():
        f, d, s = rng.choice(FORMULAS), rng.choice(DISEASES), rng.choice(SYNDROMES)
        a, b, c = rng.sample(SYMPTOMS, 3)
        return (
            f"{f.zh}用于{d.zh}时，应重点观察{a.zh}、{b.zh}及{c.zh}是否符合{s.zh}。",
            f"When {f.en} is used for {d.en}, attention should be paid to whether {a.en}, {b.en}, and {c.en} conform to {s.en}.",
            "clinical_observation",
            [ent(f), ent(d), ent(a), ent(b), ent(c), ent(s)],
        )

    def t14():
        h, o, s = rng.choice(HERBS), rng.choice(ORGANS), rng.choice(SYNDROMES)
        return (
            f"{h.zh}既可作用于{o.zh}相关病机，也可在{s.zh}中体现扶正祛邪的含义。",
            f"{h.en} may act on pathogenesis related to the {o.en} and may also reflect the idea of supporting healthy qi and dispelling pathogens in {s.en}.",
            "herb_function",
            [ent(h), ent(o), ent(s)],
        )

    def t15():
        d, th, h1, h2 = rng.choice(DISEASES), rng.choice(THERAPIES), rng.choice(HERBS), rng.choice(HERBS)
        return (
            f"治疗{d.zh}时，若以{th.zh}为核心，可考虑{h1.zh}配伍{h2.zh}。",
            f"When treating {d.en}, if the core method is to {th.en}, {h1.en} may be combined with {h2.en}.",
            "treatment_combination",
            [ent(d), ent(th), ent(h1), ent(h2)],
        )

    def t16():
        s, f, th = rng.choice(SYNDROMES), rng.choice(FORMULAS), rng.choice(THERAPIES)
        return (
            f"从机器翻译角度看，{s.zh}、{f.zh}和{th.zh}均属于需要显式标注的专业实体。",
            f"From the perspective of machine translation, {s.en}, {f.en}, and {th.en} are all specialized entities that require explicit annotation.",
            "mt_annotation",
            [ent(s), ent(f), ent(th)],
        )

    def t17():
        o1, o2, s = rng.sample(ORGANS, 2) + [rng.choice(SYNDROMES)]
        th = rng.choice(THERAPIES)
        return (
            f"{o1.zh}与{o2.zh}失衡可形成{s.zh}，治法上宜{th.zh}。",
            f"Imbalance between the {o1.en} and {o2.en} may lead to {s.en}, and the treatment method should {th.en}.",
            "organ_relation",
            [ent(o1), ent(o2), ent(s), ent(th)],
        )

    def t18():
        h, a, s = rng.choice(HERBS), rng.choice(SYMPTOMS), rng.choice(SYNDROMES)
        return (
            f"若原文中{h.zh}与{a.zh}距离较远，模型仍需判断其与{s.zh}的语义关系。",
            f"If {h.en} and {a.en} are far apart in the source text, the model still needs to identify their semantic relation to {s.en}.",
            "long_distance_dependency",
            [ent(h), ent(a), ent(s)],
        )

    def t19():
        f, h, s, d = rng.choice(FORMULAS), rng.choice(HERBS), rng.choice(SYNDROMES), rng.choice(DISEASES)
        return (
            f"{d.zh}医案中提到{f.zh}加{h.zh}，其辨证基础多为{s.zh}。",
            f"In a medical case of {d.en}, {f.en} with added {h.en} was mentioned, and the differentiation basis was mostly {s.en}.",
            "case_record",
            [ent(d), ent(f), ent(h), ent(s)],
        )

    def t20():
        s, th, a, b = rng.choice(SYNDROMES), rng.choice(THERAPIES), rng.choice(SYMPTOMS), rng.choice(SYMPTOMS)
        return (
            f"{s.zh}句子若误译为普通症状描述，会削弱{th.zh}的专业含义，并影响{a.zh}、{b.zh}的解释。",
            f"If a sentence containing {s.en} is mistranslated as a general symptom description, it weakens the specialized meaning of {th.en} and affects the interpretation of {a.en} and {b.en}.",
            "error_analysis",
            [ent(s), ent(th), ent(a), ent(b)],
        )

    return [
        t1,
        t2,
        t3,
        t4,
        t5,
        t6,
        t7,
        t8,
        t9,
        t10,
        t11,
        t12,
        t13,
        t14,
        t15,
        t16,
        t17,
        t18,
        t19,
        t20,
    ]


def generate_rows() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    templates = make_templates(rng)
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    attempts = 0
    while len(rows) < TOTAL and attempts < TOTAL * 50:
        attempts += 1
        fn = templates[attempts % len(templates)]
        zh, en, text_type, entities = fn()
        if zh in seen:
            continue
        seen.add(zh)
        rows.append(
            {
                "id": f"TCM_SYN_{len(rows) + 1:04d}",
                "zh": zh,
                "en": en,
                "text_type": text_type,
                "entities": entities,
                "term_count": len(entities),
                "source_status": "synthetic_placeholder",
                "note": "Generated for rapid experiment scaffolding; replace or disclose before submission.",
            }
        )
    if len(rows) != TOTAL:
        raise RuntimeError(f"Generated {len(rows)} unique rows, expected {TOTAL}")
    rng.shuffle(rows)
    for idx, row in enumerate(rows, 1):
        row["id"] = f"TCM_SYN_{idx:04d}"
        if idx <= TRAIN_N:
            row["split"] = "train"
        elif idx <= TRAIN_N + DEV_N:
            row["split"] = "dev"
        else:
            row["split"] = "test"
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key, "")
                if key == "entities":
                    value = json.dumps(value, ensure_ascii=False)
                out[key] = value
            writer.writerow(out)


def write_terms(path: Path) -> None:
    rows = []
    for idx, term in enumerate(TERMS, 1):
        rows.append(
            {
                "term_id": f"TERM_{idx:03d}",
                "zh": term.zh,
                "en": term.en,
                "entity_type": term.label,
                "aliases": "; ".join(term.aliases),
                "source_status": "draft_translation_needs_domain_review",
                "note": "Draft controlled vocabulary for experiments; verify against the final terminology standard.",
            }
        )
    write_csv(
        path,
        rows,
        ["term_id", "zh", "en", "entity_type", "aliases", "source_status", "note"],
    )


def entity_spans(text: str) -> list[tuple[int, int, Term]]:
    sorted_terms = sorted(TERMS, key=lambda t: len(t.zh), reverse=True)
    spans: list[tuple[int, int, Term]] = []
    occupied = [False] * len(text)
    for term in sorted_terms:
        start = 0
        while True:
            idx = text.find(term.zh, start)
            if idx < 0:
                break
            end = idx + len(term.zh)
            if not any(occupied[idx:end]):
                spans.append((idx, end, term))
                for pos in range(idx, end):
                    occupied[pos] = True
            start = idx + 1
    return sorted(spans, key=lambda x: x[0])


def write_bio(path: Path, rows: list[dict[str, object]], n: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows[:n]:
        text = str(row["zh"])
        labels = ["O"] * len(text)
        for start, end, term in entity_spans(text):
            labels[start] = f"B-{term.label}"
            for pos in range(start + 1, end):
                labels[pos] = f"I-{term.label}"
        lines.append(f"# sent_id = {row['id']}")
        lines.append(f"# split = {row['split']}")
        lines.append(f"# text = {text}")
        for char, label in zip(text, labels):
            if char.strip():
                lines.append(f"{char} {label}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def write_bio_preview(path: Path, rows: list[dict[str, object]], n: int = 60) -> None:
    preview_rows = []
    for row in rows[:n]:
        spans = [
            {"start": start, "end": end, "text": str(row["zh"])[start:end], "type": term.label}
            for start, end, term in entity_spans(str(row["zh"]))
        ]
        preview_rows.append(
            {
                "id": row["id"],
                "split": row["split"],
                "zh": row["zh"],
                "spans_json": json.dumps(spans, ensure_ascii=False),
            }
        )
    write_csv(path, preview_rows, ["id", "split", "zh", "spans_json"])


def write_evaluation_templates(base: Path, rows: list[dict[str, object]]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    main_rows = [
        {
            "model_id": "B1_general_translation",
            "description": "General translation model without TCM adaptation",
            "BLEU": "",
            "chrF": "",
            "TER": "",
            "TA": "",
            "TCR": "",
            "notes": "Baseline 1",
        },
        {
            "model_id": "B2_domain_adapted",
            "description": "General model fine-tuned on the TCM training set",
            "BLEU": "",
            "chrF": "",
            "TER": "",
            "TA": "",
            "TCR": "",
            "notes": "Baseline 2",
        },
        {
            "model_id": "B3_entity_aware",
            "description": "Entity-tagged input without domain fine-tuning",
            "BLEU": "",
            "chrF": "",
            "TER": "",
            "TA": "",
            "TCR": "",
            "notes": "Baseline 3",
        },
        {
            "model_id": "Ours_entity_aware_domain_adapted",
            "description": "Entity-aware translation with domain adaptation",
            "BLEU": "",
            "chrF": "",
            "TER": "",
            "TA": "",
            "TCR": "",
            "notes": "Full method",
        },
    ]
    write_csv(
        base / "main_results_template.csv",
        main_rows,
        ["model_id", "description", "BLEU", "chrF", "TER", "TA", "TCR", "notes"],
    )

    ablation_rows = [
        {
            "setting": "Full method",
            "removed_component": "None",
            "BLEU": "",
            "chrF": "",
            "TER": "",
            "TA": "",
            "TCR": "",
            "expected_observation": "Best or near-best overall performance",
        },
        {
            "setting": "w/o entity enhancement",
            "removed_component": "Entity tags / entity-aware input",
            "BLEU": "",
            "chrF": "",
            "TER": "",
            "TA": "",
            "TCR": "",
            "expected_observation": "Terminology accuracy should drop",
        },
        {
            "setting": "w/o domain adaptation",
            "removed_component": "Fine-tuning on TCM parallel corpus",
            "BLEU": "",
            "chrF": "",
            "TER": "",
            "TA": "",
            "TCR": "",
            "expected_observation": "General fluency may remain but domain terms should be weaker",
        },
        {
            "setting": "w/o terminology normalization",
            "removed_component": "Post-processing / controlled vocabulary normalization",
            "BLEU": "",
            "chrF": "",
            "TER": "",
            "TA": "",
            "TCR": "",
            "expected_observation": "Consistency should drop more than BLEU",
        },
    ]
    write_csv(
        base / "ablation_template.csv",
        ablation_rows,
        ["setting", "removed_component", "BLEU", "chrF", "TER", "TA", "TCR", "expected_observation"],
    )

    test_rows = [r for r in rows if r["split"] == "test"][:50]
    human_rows = []
    for row in test_rows:
        human_rows.append(
            {
                "id": row["id"],
                "zh": row["zh"],
                "reference_en": row["en"],
                "model_output": "",
                "accuracy_1_5": "",
                "readability_1_5": "",
                "terminology_normativity_1_5": "",
                "professional_consistency_1_5": "",
                "rater": "",
                "comment": "",
            }
        )
    write_csv(
        base / "human_evaluation_template.csv",
        human_rows,
        [
            "id",
            "zh",
            "reference_en",
            "model_output",
            "accuracy_1_5",
            "readability_1_5",
            "terminology_normativity_1_5",
            "professional_consistency_1_5",
            "rater",
            "comment",
        ],
    )

    case_rows = []
    for row in [r for r in rows if r["split"] == "test"][:12]:
        case_rows.append(
            {
                "id": row["id"],
                "zh": row["zh"],
                "reference_en": row["en"],
                "baseline_output": "",
                "ours_output": "",
                "key_entities": json.dumps(row["entities"], ensure_ascii=False),
                "error_type": "",
                "analysis": "",
            }
        )
    write_csv(
        base / "case_analysis_template.csv",
        case_rows,
        ["id", "zh", "reference_en", "baseline_output", "ours_output", "key_entities", "error_type", "analysis"],
    )


def write_markdown_docs(base: Path, rows: list[dict[str, object]]) -> None:
    docs = base / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": sum(1 for r in rows if r["split"] == "train"),
        "dev": sum(1 for r in rows if r["split"] == "dev"),
        "test": sum(1 for r in rows if r["split"] == "test"),
    }
    type_counts: dict[str, int] = {}
    for term in TERMS:
        type_counts[term.label] = type_counts.get(term.label, 0) + 1

    (base / "README.md").write_text(
        f"""# TCM Translation Experiment Materials

This folder contains a rapid experiment scaffold for an entity-aware TCM machine translation paper.

Important: the parallel corpus and terminology translations are synthetic/draft materials generated for pipeline development. They must be replaced by verified data or explicitly disclosed as synthetic data before any academic submission.

## Files

- `data/full_corpus.csv`: all {TOTAL} parallel sentence pairs.
- `data/train.csv`: {counts["train"]} training pairs.
- `data/dev.csv`: {counts["dev"]} validation pairs.
- `data/test.csv`: {counts["test"]} test pairs.
- `terminology/terminology.csv`: controlled vocabulary with entity labels.
- `bio/bio_samples.conll`: character-level BIO labels for 300 sample sentences.
- `bio/bio_samples_preview.csv`: span preview for quick checking.
- `evaluation/*.csv`: result, ablation, human evaluation, and case analysis templates.
- `docs/data_statement.md`: dataset disclaimer and usage notes.
- `docs/experiment_plan.md`: three-day experiment plan and minimal reporting checklist.

## Split

The split ratio is 7:1.5:1.5:

- Train: {counts["train"]}
- Dev: {counts["dev"]}
- Test: {counts["test"]}

Random seed: {SEED}
""",
        encoding="utf-8-sig",
    )

    (docs / "data_statement.md").write_text(
        f"""# Data Statement

## Dataset Status

This dataset is a synthetic placeholder corpus generated on 2026-04-27 for rapid experiment scaffolding. It is suitable for debugging the translation pipeline, metric scripts, entity tagging logic, and table formats.

It is not a verified clinical, textbook, guideline, or journal corpus. Do not describe it as real collected data unless it is replaced by YYQ's verified source data.

## Corpus Size and Split

- Total sentence pairs: {TOTAL}
- Train/dev/test: {counts["train"]}/{counts["dev"]}/{counts["test"]}
- Split ratio: 7/1.5/1.5
- Generation seed: {SEED}

## Labels

The controlled vocabulary includes these entity categories:

{chr(10).join(f"- {k}: {v} terms" for k, v in sorted(type_counts.items()))}

## Recommended Use

Use these files to implement:

1. General translation baseline.
2. Domain-adapted translation model.
3. Entity-aware translation input.
4. Entity-aware domain-adapted model.
5. BLEU, chrF/TER, TA, TCR, human evaluation, and case analysis.

For the final paper, replace this dataset with sourced parallel text, or add a clear synthetic-data section explaining the generation method and limitations.
""",
        encoding="utf-8-sig",
    )

    (docs / "entity_label_schema.md").write_text(
        """# Entity Label Schema

The BIO file uses character-level labels.

- `B-HERB` / `I-HERB`: Chinese materia medica terms.
- `B-FORMULA` / `I-FORMULA`: prescription or formula names.
- `B-SYNDROME` / `I-SYNDROME`: TCM pattern or syndrome names.
- `B-THERAPY` / `I-THERAPY`: treatment principles or methods.
- `B-ORGAN` / `I-ORGAN`: zang-fu organs and related body-system terms.
- `B-SYMPTOM` / `I-SYMPTOM`: clinical manifestations.
- `B-DISEASE` / `I-DISEASE`: disease or topic names.
- `O`: non-entity character.

Longest-match labeling is used when terms overlap. For example, `肾阴虚证` is labeled as a single `SYNDROME` span rather than splitting `肾` as `ORGAN`.
""",
        encoding="utf-8-sig",
    )

    (docs / "experiment_plan.md").write_text(
        """# Minimal Experiment Plan

## Research Questions

1. Does domain adaptation improve translation quality for low-resource TCM text?
2. Does entity-aware input improve terminology accuracy and consistency?
3. Does combining entity enhancement and domain adaptation outperform either component alone?

## Model Settings

- B1: General translation model.
- B2: Domain-adapted model fine-tuned on `train.csv`.
- B3: Entity-aware model using tagged source input.
- Ours: Entity-aware input plus domain adaptation.

## Metrics

- BLEU: overall n-gram translation quality.
- chrF or TER: character-level or edit-distance complement.
- TA: terminology accuracy, calculated against `terminology.csv`.
- TCR: terminology consistency rate across repeated terms.
- Human evaluation: accuracy, readability, terminology normativity, professional consistency.

## Three-Day Work Plan

Day 1:
- Freeze dataset format and entity labels.
- Run baseline translation.
- Build entity-tagging script and metric scripts.

Day 2:
- Run domain adaptation and entity-aware experiments.
- Compute automatic metrics.
- Select 30-50 human evaluation samples.

Day 3:
- Finish ablation results.
- Fill case analysis table.
- Write experiment section, limitations, and final tables.
""",
        encoding="utf-8-sig",
    )

    config = {
        "seed": SEED,
        "total": TOTAL,
        "split": {"train": counts["train"], "dev": counts["dev"], "test": counts["test"]},
        "entity_labels": sorted(type_counts),
        "metrics": ["BLEU", "chrF", "TER", "TA", "TCR", "human_eval"],
        "disclaimer": "Synthetic placeholder dataset. Replace or disclose before submission.",
    }
    (base / "experiment_config_template.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out = repo / "generated_materials"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = generate_rows()
    fields = ["id", "split", "zh", "en", "text_type", "term_count", "entities", "source_status", "note"]
    write_csv(out / "data" / "full_corpus.csv", rows, fields)
    write_csv(out / "data" / "train.csv", [r for r in rows if r["split"] == "train"], fields)
    write_csv(out / "data" / "dev.csv", [r for r in rows if r["split"] == "dev"], fields)
    write_csv(out / "data" / "test.csv", [r for r in rows if r["split"] == "test"], fields)
    write_terms(out / "terminology" / "terminology.csv")
    write_bio(out / "bio" / "bio_samples.conll", rows)
    write_bio_preview(out / "bio" / "bio_samples_preview.csv", rows)
    write_evaluation_templates(out / "evaluation", rows)
    write_markdown_docs(out, rows)

    summary = {
        "output_dir": str(out),
        "total": len(rows),
        "train": sum(1 for r in rows if r["split"] == "train"),
        "dev": sum(1 for r in rows if r["split"] == "dev"),
        "test": sum(1 for r in rows if r["split"] == "test"),
        "terms": len(TERMS),
        "bio_sentences": 300,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
