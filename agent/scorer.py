import re


def extract_score(review_text: str) -> int:
    """
    Parse the LLM review response and extract the numeric score.
    Looks for patterns like 'SCORE: 87/100' or 'Score: 87/100'.
    Returns 0 if no score is found.
    """
    patterns = [
        r'SCORE:\s*(\d+)\s*/\s*100',
        r'Score:\s*(\d+)\s*/\s*100',
        r'score:\s*(\d+)\s*/\s*100',
        r'\*\*Score[:\s]+(\d+)/100\*\*',
        r'\*\*SCORE[:\s]+(\d+)/100\*\*',
    ]
    for pattern in patterns:
        match = re.search(pattern, review_text)
        if match:
            score = int(match.group(1))
            return min(100, max(0, score))
    return 0


def get_certification(score: int, threshold: int = 95) -> dict:
    """
    Return a certification result dict based on the score.
    """
    certified = score >= threshold
    return {
        "score": score,
        "threshold": threshold,
        "certified": certified,
        "status": "CERTIFIED" if certified else "NOT CERTIFIED",
        "badge": "✅ CERTIFIED" if certified else "❌ NOT CERTIFIED",
        "message": (
            f"Code passed QA review with a score of {score}/100 "
            f"and is certified for execution on the Cloudera platform."
            if certified else
            f"Code scored {score}/100, below the required {threshold}/100. "
            f"Review the findings below, fix the issues, and resubmit."
        ),
    }
