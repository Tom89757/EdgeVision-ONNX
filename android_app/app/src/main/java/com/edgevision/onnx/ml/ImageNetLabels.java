package com.edgevision.onnx.ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ImageNet 1k Top-K 后处理与常用类别中文名映射。
 *
 * 索引来源为 ImageNet 2012 的 1000 类标准顺序（与 torchvision 预训练权重输出一致），
 * 这里仅内置约 40 个家居、宠物和人物相关类别；其余类别显示 class_{index}。
 */
public final class ImageNetLabels {
    private static final Map<Integer, String> CHINESE_NAMES;

    static {
        Map<Integer, String> names = new HashMap<>();

        // 犬类全量（ImageNet 索引 151-268）。
        names.put(151, "吉娃娃");
        names.put(152, "日本狆");
        names.put(153, "马尔济斯犬");
        names.put(154, "北京犬");
        names.put(155, "西施犬");
        names.put(156, "布伦海姆猎犬");
        names.put(157, "蝴蝶犬");
        names.put(158, "玩具猎梗");
        names.put(159, "罗得西亚脊背犬");
        names.put(160, "阿富汗猎犬");
        names.put(161, "巴吉度猎犬");
        names.put(162, "比格犬");
        names.put(163, "寻血猎犬");
        names.put(164, "蓝斑猎犬");
        names.put(165, "黑褐浣熊猎犬");
        names.put(166, "沃克猎犬");
        names.put(167, "英国猎狐犬");
        names.put(168, "红骨猎犬");
        names.put(169, "波索尔犬");
        names.put(170, "爱尔兰猎狼犬");
        names.put(171, "意大利灵缇");
        names.put(172, "惠比特犬");
        names.put(173, "伊比赞猎犬");
        names.put(174, "挪威猎麋犬");
        names.put(175, "水獭猎犬");
        names.put(176, "萨路基猎犬");
        names.put(177, "苏格兰猎鹿犬");
        names.put(178, "魏玛犬");
        names.put(179, "斯塔福郡斗牛梗");
        names.put(180, "美国斯塔福郡梗");
        names.put(181, "贝灵顿梗");
        names.put(182, "边境梗");
        names.put(183, "凯利蓝梗");
        names.put(184, "爱尔兰梗");
        names.put(185, "诺福克梗");
        names.put(186, "诺威奇梗");
        names.put(187, "约克夏梗");
        names.put(188, "硬毛猎狐梗");
        names.put(189, "湖畔梗");
        names.put(190, "西里汉梗");
        names.put(191, "艾尔谷梗");
        names.put(192, "凯恩梗");
        names.put(193, "澳洲梗");
        names.put(194, "丹迪丁蒙梗");
        names.put(195, "波士顿斗牛犬");
        names.put(196, "迷你雪纳瑞");
        names.put(197, "巨型雪纳瑞");
        names.put(198, "标准雪纳瑞");
        names.put(199, "苏格兰梗");
        names.put(200, "西藏梗");
        names.put(201, "丝毛梗");
        names.put(202, "软毛麦色梗");
        names.put(203, "西高地白梗");
        names.put(204, "拉萨犬");
        names.put(205, "平毛寻回犬");
        names.put(206, "卷毛寻回犬");
        names.put(207, "金毛寻回犬");
        names.put(208, "拉布拉多寻回犬");
        names.put(209, "切萨皮克湾寻回犬");
        names.put(210, "德国短毛指示犬");
        names.put(211, "维兹拉犬");
        names.put(212, "英国雪达犬");
        names.put(213, "爱尔兰雪达犬");
        names.put(214, "戈登雪达犬");
        names.put(215, "布列塔尼猎犬");
        names.put(216, "克伦伯猎犬");
        names.put(217, "英国激飞猎犬");
        names.put(218, "威尔士激飞猎犬");
        names.put(219, "可卡犬");
        names.put(220, "萨塞克斯猎犬");
        names.put(221, "爱尔兰水猎犬");
        names.put(222, "库瓦兹犬");
        names.put(223, "西帕基犬");
        names.put(224, "格罗安达犬");
        names.put(225, "马里努阿犬");
        names.put(226, "布里牧羊犬");
        names.put(227, "凯尔皮犬");
        names.put(228, "可蒙犬");
        names.put(229, "古英国牧羊犬");
        names.put(230, "喜乐蒂牧羊犬");
        names.put(231, "柯利牧羊犬");
        names.put(232, "边境牧羊犬");
        names.put(233, "法兰德斯畜牧犬");
        names.put(234, "罗威纳犬");
        names.put(235, "德国牧羊犬");
        names.put(236, "杜宾犬");
        names.put(237, "迷你杜宾犬");
        names.put(238, "大瑞士山地犬");
        names.put(239, "伯恩山犬");
        names.put(240, "阿彭策尔山地犬");
        names.put(241, "恩特雷布赫山地犬");
        names.put(242, "拳师犬");
        names.put(243, "斗牛獒犬");
        names.put(244, "藏獒");
        names.put(245, "法国斗牛犬");
        names.put(246, "大丹犬");
        names.put(247, "圣伯纳犬");
        names.put(248, "爱斯基摩犬");
        names.put(249, "阿拉斯加雪橇犬");
        names.put(250, "西伯利亚哈士奇");
        names.put(251, "大麦町犬");
        names.put(252, "猴面犬");
        names.put(253, "巴森吉犬");
        names.put(254, "巴哥犬");
        names.put(255, "兰波格犬");
        names.put(256, "纽芬兰犬");
        names.put(257, "大白熊犬");
        names.put(258, "萨摩耶犬");
        names.put(259, "博美犬");
        names.put(260, "松狮犬");
        names.put(261, "荷兰毛狮犬");
        names.put(262, "布拉班特格里芬犬");
        names.put(263, "彭布罗克威尔士柯基");
        names.put(264, "卡迪根威尔士柯基");
        names.put(265, "玩具贵宾犬");
        names.put(266, "迷你贵宾犬");
        names.put(267, "标准贵宾犬");
        names.put(268, "墨西哥无毛犬");

        // 猫科（ImageNet 索引 281-285）。
        names.put(281, "虎斑猫");
        names.put(282, "虎纹猫");
        names.put(283, "波斯猫");
        names.put(284, "暹罗猫");
        names.put(285, "埃及猫");

        // 常见家居/室内物品。
        names.put(503, "咖啡杯");
        names.put(507, "电脑键盘");
        names.put(525, "书桌");
        names.put(526, "台式电脑");
        names.put(531, "餐桌");
        names.put(618, "笔记本电脑");
        names.put(664, "显示器");
        names.put(672, "鼠标");
        names.put(721, "枕头");
        names.put(761, "冰箱");
        names.put(846, "台灯");
        names.put(851, "电视机");
        names.put(861, "马桶圈");

        // 人物相关类别。
        names.put(981, "球类运动员");
        names.put(982, "新郎");
        names.put(983, "水肺潜水员");

        CHINESE_NAMES = Collections.unmodifiableMap(names);
    }

    private ImageNetLabels() {
    }

    public static final class Prediction {
        private final int classIndex;
        private final float confidence;
        private final String chineseName;

        private Prediction(int classIndex, float confidence, String chineseName) {
            this.classIndex = classIndex;
            this.confidence = confidence;
            this.chineseName = chineseName;
        }

        public int getClassIndex() {
            return classIndex;
        }

        public float getConfidence() {
            return confidence;
        }

        public String getChineseName() {
            return chineseName;
        }
    }

    public static List<Prediction> topK(float[] logits, int requestedK) {
        if (logits == null || logits.length == 0) {
            return Collections.emptyList();
        }
        int k = Math.max(1, Math.min(requestedK, logits.length));
        float[] probabilities = softmax(logits);
        Prediction[] top = new Prediction[k];

        for (int index = 0; index < probabilities.length; index++) {
            float probability = probabilities[index];
            int insertAt = -1;
            for (int rank = 0; rank < k; rank++) {
                if (top[rank] == null || probability > top[rank].getConfidence()) {
                    insertAt = rank;
                    break;
                }
            }
            if (insertAt < 0) {
                continue;
            }
            for (int rank = k - 1; rank > insertAt; rank--) {
                top[rank] = top[rank - 1];
            }
            top[insertAt] = new Prediction(index, probability, getChineseName(index));
        }

        List<Prediction> result = new ArrayList<>(k);
        for (Prediction prediction : top) {
            if (prediction != null) {
                result.add(prediction);
            }
        }
        return result;
    }

    public static String getChineseName(int classIndex) {
        String name = CHINESE_NAMES.get(classIndex);
        return name == null ? "class_" + classIndex : name;
    }

    /** 判断是否为 ImageNet 中的犬类或家猫类别。 */
    public static boolean isPetClass(int classIndex) {
        return (classIndex >= 151 && classIndex <= 268)
                || (classIndex >= 281 && classIndex <= 285);
    }

    private static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            if (!Float.isNaN(logit) && logit > max) {
                max = logit;
            }
        }
        if (!Float.isFinite(max)) {
            max = 0.0f;
        }

        double sum = 0.0;
        float[] probabilities = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            float logit = logits[i];
            float probability = Float.isNaN(logit) ? 0.0f : (float) Math.exp(logit - max);
            probabilities[i] = probability;
            sum += probability;
        }
        if (sum <= 0.0) {
            return probabilities;
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = (float) (probabilities[i] / sum);
        }
        return probabilities;
    }
}
