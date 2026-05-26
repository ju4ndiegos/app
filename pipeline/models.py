from django.db import models


MALWARE_CLASSES = [
    ("adware",   "Adware"),
    ("banking",  "Banking"),
    ("benign",   "Benign"),
    ("riskware", "Riskware"),
    ("sms",      "SMS Malware"),
    ("unknown",  "Unknown"),
]


class AnalysisResult(models.Model):
    apk_hash         = models.CharField(max_length=64, unique=True)
    label            = models.CharField(max_length=32, choices=MALWARE_CLASSES, blank=True)
    image            = models.ImageField(upload_to="results/")
    sequence_preview = models.TextField()
    tabular_json     = models.JSONField()
    prediction_json  = models.JSONField(null=True, blank=True)  # {label, index, probs}
    created_at       = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.apk_hash[:16]}… ({self.label})"

    def active_features(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "permissions": [],
            "actions": [],
            "categories": [],
            "services": [],
        }
        for col, val in self.tabular_json.items():
            if val != 1:
                continue
            if col.startswith("permission."):
                groups["permissions"].append(col[len("permission."):])
            elif col.startswith("action."):
                groups["actions"].append(col[len("action."):])
            elif col.startswith("category."):
                groups["categories"].append(col[len("category."):])
            else:
                groups["services"].append(col)
        return groups
