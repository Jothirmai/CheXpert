import { Link, useLocation } from 'react-router-dom'
import { LayoutDashboard, FileText, ShieldAlert } from 'lucide-react'

export default function HomePage() {
  const location = useLocation()
  const active = location.pathname === '/' || location.pathname.startsWith('/history')
  return (
    <div className="max-w-5xl mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-900">Chest Xpert</h1>
        <p className="text-slate-600 mt-1">
          AI-assisted chest X-ray analysis using pre-trained deep learning models.
          Upload an image to receive a structured radiology report draft.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
          <div className="w-10 h-10 rounded bg-blue-100 flex items-center justify-center text-blue-600 mb-4">
            <UploadIcon className="w-5 h-5" />
          </div>
          <h3 className="font-semibold text-slate-900 mb-2">Upload Image</h3>
          <p className="text-sm text-slate-600 mb-4">
            Select a chest X-ray (PNG, JPG, JPEG, DICOM) to begin analysis.
          </p>
          <Link to="/upload" className="inline-flex items-center gap-2 text-sm font-medium text-blue-600 hover:underline">
            Start Analysis <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
          <div className="w-10 h-10 rounded bg-green-100 flex items-center justify-center text-green-600 mb-4">
            <FileText className="w-5 h-5" />
          </div>
          <h3 className="font-semibold text-slate-900 mb-2">Structured Report</h3>
          <p className="text-sm text-slate-600 mb-4">
            View 14 CNN findings, the report draft, and reviewer evidence.
          </p>
          <Link to="/history" className="inline-flex items-center gap-2 text-sm font-medium text-blue-600 hover:underline">
            View Reports <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
          <div className="w-10 h-10 rounded bg-purple-100 flex items-center justify-center text-purple-600 mb-4">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <h3 className="font-semibold text-slate-900 mb-2">Research Use Only</h3>
          <p className="text-sm text-slate-600">
            Reports require review by a qualified radiologist before clinical use.
          </p>
        </div>
      </div>

      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
        <h3 className="font-semibold text-slate-900 mb-4">14 CNN Findings</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
          {[
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding',
          ].map((f) => (
            <span key={f} className="px-3 py-1.5 rounded bg-slate-50 text-slate-700 text-xs border border-slate-100">
              {f}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-8 bg-amber-50 border border-amber-200 rounded-lg p-4">
        <p className="text-sm text-amber-800">
          <strong>DISCLAIMER:</strong> AI-generated research draft requiring review by a qualified radiologist.
          Not a final clinical diagnosis. Prediction scores are not calibrated probabilities.
        </p>
      </div>
    </div>
  )
}

function UploadIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}

function ArrowRight() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14" />
      <path d="M12 5l7 7-7 7" />
    </svg>
  )
}
