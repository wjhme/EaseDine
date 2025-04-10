import Levenshtein

def calculate_cer(reference: str, hypothesis: str, 
                 detail_analysis: bool = False) -> float:
    """
    计算中文字错率（CER）
    
    参数:
        reference : 标注文本（标准答案）
        hypothesis : 识别结果文本
        detail_analysis : 是否返回详细编辑操作统计
        
    返回:
        CER值（当detail=True时返回包含详细信息的字典）
    """
    
    edit_ops = Levenshtein.editops(reference, hypothesis)
    edits = {'insertions': 0, 'deletions': 0, 'substitutions': 0}
    for op in edit_ops:
        if op[0] == 'insert':
            edits['insertions'] += 1
        elif op[0] == 'delete':
            edits['deletions'] += 1
        elif op[0] == 'replace':
            edits['substitutions'] += 1
    distance = Levenshtein.distance(reference, hypothesis)
    
    # 计算总字符数（中文按字统计）
    ref_len = len(reference)
    
    # 处理空标注文本的特殊情况
    if ref_len == 0:
        cer = 1.0 if len(hypothesis) > 0 else 0.0
        if detail_analysis:
            return {'cer': cer, 'edits': edits, 'distance': distance}
        return cer
    
    cer = distance / ref_len
    
    if detail_analysis:
        return {
            'cer': cer,
            'distance': distance,
            'ref_len': ref_len,
            'edits': edits,
            'insertions': edits['insertions'],
            'deletions': edits['deletions'],
            'substitutions': edits['substitutions']
        }
    return cer

# 测试用例
if __name__ == "__main__":
    # 常规测试
    test_cases = [
        ("测试", "测试", 0.0),         # 完全正确
        ("测试", "", 2/2),            # 全错
        ("你好世界", "你好世届", 1/4),  # 1字替换
        ("语音识别", "语音识别系统", 2/4), # 多插入
        ("人工智能", "人工智", 1/4),     # 删除
        ("", "测试", 1.0),            # 空标注
        ("", "", 0.0),               # 双空
    ]
    
    for ref, hyp, expected in test_cases:
        result = calculate_cer(ref, hyp)
        print(result)
        assert abs(result - expected) < 1e-6, \
            f"测试失败: {ref} vs {hyp} | 预期 {expected} 实际 {result}"
    
    
    print("所有测试通过！")