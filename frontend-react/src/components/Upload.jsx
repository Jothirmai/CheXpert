import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, Image, AlertCircle, CheckCircle, Clock } from 'lucide-react'
import { api } from '../api'

export default function UploadPage() {
  const navigate = useNavigate()
  const [templates,setTemplates] = useState([])
  const [templateId,setTemplateId] = useState('standard')
  const [patientId,setPatientId] = useState('')
  const [indication,setIndication] = useState('')
  const [comparison,setComparison] = useState('')
  const [view,setView] = useState('')
  useEffect(()=>{api.templates().then(setTemplates).catch(e=>setError(e.message))},[])
  const [file, setFile] = useState(null)
  const [additionalFiles,setAdditionalFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)
  const [analysisId, setAnalysisId] = useState(null)
  const [stage, setStage] = useState('idle') // idle, uploading, analyzing, done, error

  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files)
    if(selected.length>5){setError('Select at most five images from the same study.');return}
    setFile(selected[0]||null);setAdditionalFiles(selected.slice(1));setError(null)
  }

  const handleUpload = async () => {
    if (!file) return
    setError(null)
    setUploading(true)
    setStage('uploading')
    setProgress(10)

    const formData = new FormData()
    formData.append('file', file)
    additionalFiles.forEach(image=>formData.append('additional_files',image))
    formData.append('generate_grad_cam', 'true')
    formData.append('patient_id', patientId)
    formData.append('clinical_indication', indication)
    formData.append('comparison_study', comparison)
    formData.append('view', view)
    formData.append('template_id', templateId)

    try {
      const result = await api.uploadAnalysis(formData)
      setProgress(60)
      setStage('analyzing')
      setAnalysisId(result.analysis_id || result.id)
      // Poll
      const analysis = await pollAnalysis(result.analysis_id || result.id)
      setProgress(100)
      setStage('done')
      navigate(`/report/${result.analysis_id || result.id}`)
    } catch (err) {
      setError(err.message)
      setStage('error')
    } finally {
      setUploading(false)
    }
  }

  const pollAnalysis = async (id) => {
    for (let i = 0; i < 150; i++) {
      await new Promise((r) => setTimeout(r, 2000))
      const res = await api.getAnalysis(id)
      if (res.status === 'completed') return res
      if (res.status === 'failed') throw new Error(res.modality?.reason || 'Analysis failed')
      setProgress(Math.min(60 + i * 0.4, 95))
    }
    throw new Error('Analysis timed out')
  }

  return (
    <div className="max-w-3xl mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-900">New Analysis</h1>
        <p className="text-slate-600 mt-1">Upload the original images from one study. Include available frontal and lateral views for a complete report.</p>
      </div>

      <label className="block my-3">Patient ID<input disabled={uploading} className="border p-2 ml-2" value={patientId} onChange={e=>setPatientId(e.target.value)}/></label>
      <label className="block my-3">Clinical indication<input disabled={uploading} className="border p-2 ml-2" value={indication} onChange={e=>setIndication(e.target.value)}/></label>
      <label className="block my-3">Comparison study<input disabled={uploading} className="border p-2 ml-2" value={comparison} onChange={e=>setComparison(e.target.value)}/></label>
      <label className="block my-3">Projection<select disabled={uploading} className="border p-2 ml-2" value={view} onChange={e=>setView(e.target.value)}><option value="">Not provided</option><option value="PA">PA</option><option value="AP">AP</option><option value="Lateral">Lateral</option><option value="PA and lateral">PA and lateral</option><option value="AP and lateral">AP and lateral</option></select></label>
      <label className="block my-3">Report template<select disabled={uploading} className="border p-2 ml-2" value={templateId} onChange={e=>setTemplateId(e.target.value)}>{templates.map(t=><option key={t.id} value={t.id}>{t.template_name}</option>)}</select></label>
      <div className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm">
        <div className="flex flex-col items-center gap-4 mb-6">
          <div className="w-full min-h-40 rounded-xl border-2 border-dashed border-slate-300 flex items-center justify-center bg-slate-50 hover:bg-blue-50 transition cursor-pointer group relative px-6 py-5 text-center">
            <input disabled={uploading} type="file" multiple accept="image/*,.dcm,.dicom" className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" onChange={handleFileChange} />
            {file ? (
              <div className="flex flex-col items-center gap-2">
                <Image className="w-10 h-10 text-slate-400" />
                <p className="text-sm font-medium text-slate-700">{file.name}{additionalFiles.length>0?` + ${additionalFiles.length} additional view(s)`:''}</p>
                <p className="text-xs text-slate-500">Change file</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2">
                <Upload className="w-10 h-10 text-slate-400 group-hover:text-blue-600 transition" />
                <p className="text-sm font-medium text-slate-700">Upload chest X-ray</p>
                <p className="text-xs text-slate-500">PNG, JPG, JPEG or DICOM (max 50 MB)</p>
              </div>
            )}
          </div>
          {uploading && (
            <div className="flex flex-col items-center gap-2">
              <Clock className="w-6 h-6 animate-spin text-blue-600" />
              <p className="text-sm text-slate-600">{stage === 'uploading' ? 'Uploading…' : 'Checking chest X-ray, preparing CNN evidence and generating report…'}</p>
              <div className="w-48 bg-slate-100 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}
        </div>

        {error && (
          <div role="alert" className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <AlertCircle className="w-5 h-5 text-red-600 mr-2" />
            <span className="text-red-700 text-sm">{error}</span>
          </div>
        )}

        {analysisId && stage === 'done' && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6 flex items-center gap-3">
            <CheckCircle className="w-5 h-5 text-green-600" />
            <div>
              <p className="font-medium text-green-800">Analysis complete</p>
              <p className="text-xs text-green-600">Report ID: {analysisId}</p>
            </div>
          </div>
        )}

        {!uploading && stage !== 'done' && (
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="w-full py-3 px-6 bg-blue-600 text-white font-medium rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
          >
            <Upload className="w-4 h-4" /> {file ? 'Start Analysis' : 'Select an image'}
          </button>
        )}
      </div>

      <p className="my-3 text-sm">Target: about two minutes after models are loaded. Processing time depends on GPU availability. Progress is approximate.</p>
      <div className="mt-6 bg-amber-50 border border-amber-200 rounded-lg p-4">
        <p className="text-sm text-amber-800">
          <strong>Research use only.</strong> Reports are AI-generated drafts requiring review by a qualified radiologist.
        </p>
      </div>
    </div>
  )
}
