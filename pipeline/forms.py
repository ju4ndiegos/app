from django import forms
from pipeline.models import MALWARE_CLASSES


class APKUploadForm(forms.Form):
    apk_file = forms.FileField(
        label="APK file",
        help_text="Select an Android .apk file (max 100 MB)",
    )
    label = forms.ChoiceField(
        choices=MALWARE_CLASSES,
        initial="unknown",
        label="Known class (optional)",
    )
