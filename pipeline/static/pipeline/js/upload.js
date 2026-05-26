function featuresZone() {
  return {
    imgName: '', seqName: '', csvName: '',
    uploading: false,
    error: '',

    pick(e, field) {
      const f = e.target.files[0];
      if (!f) return;
      this[field] = f.name;
      this.error = '';
    },

    handleSubmit(e) {
      if (!this.imgName || !this.seqName || !this.csvName) {
        this.error = 'Please select all three files before submitting.';
        return;
      }
      this.uploading = true;
      this.error = '';
      e.target.submit();
    },

    _fmt(b) {
      if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
      if (b >= 1024)    return (b / 1024).toFixed(0) + ' KB';
      return b + ' B';
    },
  };
}

function uploadZone() {
  return {
    state: 'idle',   // 'idle' | 'hover' | 'selected' | 'uploading'
    fileName: '',
    fileSize: '',
    error: '',
    _stageIdx: 0,
    _stageTimer: null,
    _stages: [
      'Parsing APK structure…',
      'Extracting DEX image…',
      'Analyzing API sequences…',
      'Running ensemble models…',
    ],

    get currentStage() {
      return this._stages[this._stageIdx];
    },

    handleFile(e) {
      const f = e.target.files[0];
      if (!f) return;
      if (!f.name.toLowerCase().endsWith('.apk')) {
        this.error = 'Please select an Android APK file (.apk)';
        this.state = 'idle';
        return;
      }
      this.error = '';
      this.state = 'selected';
      this.fileName = f.name;
      this.fileSize = this._fmt(f.size);
    },

    handleDrop(e) {
      const files = e.dataTransfer ? e.dataTransfer.files : null;
      if (!files || !files.length) return;
      const input = document.getElementById('id_apk_file');
      if (!input) return;
      try {
        const dt = new DataTransfer();
        dt.items.add(files[0]);
        input.files = dt.files;
      } catch (_) {
        // DataTransfer not available in some environments — silent fallback
      }
      input.dispatchEvent(new Event('change'));
    },

    handleSubmit(e) {
      // Alpine's @submit.prevent already called preventDefault
      if (this.state !== 'selected') {
        this.error = 'Please select an APK file first.';
        return;
      }
      this.state = 'uploading';
      this.error = '';
      this._stageIdx = 0;
      this._stageTimer = setInterval(() => {
        this._stageIdx = (this._stageIdx + 1) % this._stages.length;
      }, 6000);
      // Submit the form natively (bypasses event listeners, includes CSRF token)
      e.target.submit();
    },

    _fmt(b) {
      if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
      if (b >= 1024)    return (b / 1024).toFixed(0) + ' KB';
      return b + ' B';
    }
  };
}
