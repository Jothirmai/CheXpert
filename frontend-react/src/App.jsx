import { useState } from 'react'
import { Home, Upload, FileText, History, Clock, Info } from 'lucide-react'
import { Link, useLocation, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import HomePage from './components/Home'
import UploadPage from './components/Upload'
import ReportPage from './components/Report'
import TemplatesPage from './components/Template'
import HistoryPage from './components/History'

const PAGES = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/upload', label: 'New Analysis', icon: Upload },
  { to: '/report', label: 'Report Review', icon: FileText },
  { to: '/history', label: 'History', icon: History },
  { to: '/templates', label: 'Templates', icon: Clock },
]

export default function App() {
  const location = useLocation()
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <Navbar />

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        className={`fixed top-14 bottom-0 left-0 w-64 bg-white border-r border-slate-200 z-40 transform translate-x-0 md:fixed md:translate-x-0 transition-transform duration-300 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="p-6 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded bg-blue-600 flex items-center justify-center text-white font-bold">
              CX
            </div>
            <span className="font-bold text-slate-800 text-sm tracking-tight">Chest Xpert</span>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="mt-4 p-2 hover:bg-slate-100 rounded-lg text-slate-600"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <nav className="p-4 space-y-1">
          {PAGES.map((item) => {
            const Icon = item.icon
            const active = location.pathname === item.to
            return (
              <Link
                key={item.to}
                to={item.to}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition ${
                  active
                    ? 'bg-blue-50 text-blue-600'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                <Icon size={18} />
                {item.label}
              </Link>
            )
          })}
        </nav>
        <div className="mt-auto p-6 border-t border-slate-100">
          <Link to="/about" className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-100 transition">
            <Info size={18} /> About
          </Link>
        </div>
      </aside>

      <main className="md:pl-64 pt-14">
        <button
          className="md:hidden p-2 mr-4 text-slate-600 hover:bg-slate-100 rounded-lg"
          onClick={() => setSidebarOpen(true)}
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/report/:id" element={<ReportPage />} />
          <Route path="/templates" element={<TemplatesPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/report" element={<HistoryPage />} />
          <Route path="/about" element={<div className="p-8">Chest Xpert assists radiologists with editable AI-generated chest X-ray reports. Human review is required.</div>} />
        </Routes>
      </main>
    </div>
  )
}
