import { Link, useLocation } from 'react-router-dom'
import { LayoutDashboard, Upload, FileText, History, Clock, Info } from 'lucide-react'

const NAV = [
  { to: '/', label: 'Home', icon: LayoutDashboard },
  { to: '/upload', label: 'New Analysis', icon: Upload },
  { to: '/report', label: 'Report Review', icon: FileText },
  { to: '/history', label: 'History', icon: History },
  { to: '/templates', label: 'Templates', icon: Clock },
]

export default function Navbar() {
  const location = useLocation()
  return (
    <nav className="fixed top-0 left-0 right-0 h-14 bg-white border-b border-slate-200 z-50">
      <div className="max-w-7xl mx-auto px-4 h-full flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-blue-600 flex items-center justify-center text-white font-bold">
            CX
          </div>
          <span className="font-bold text-slate-800 tracking-tight">Chest Xpert</span>
        </div>
        <div className="hidden md:flex items-center gap-1">
          {NAV.map((item) => {
            const Icon = item.icon
            const active = location.pathname === item.to || location.pathname.startsWith(item.to + '/')
            return (
              <Link
                key={item.to}
                to={item.to}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition ${
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
        </div>
        <div className="flex items-center gap-3">
          <button onClick={()=>{const key=window.prompt('Enter your assigned access key (leave blank to sign out)');if(key!==null){if(key.trim())localStorage.setItem('chestxpert_api_key',key.trim());else localStorage.removeItem('chestxpert_api_key');window.location.reload()}}} className="px-3 py-2 border rounded">Access key</button>
          <Link to="/about" className="px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100 rounded-lg">
            About
          </Link>

        </div>
      </div>
    </nav>
  )
}
