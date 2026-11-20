import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
export default function History(){const [items,setItems]=useState([]),[error,setError]=useState('');useEffect(()=>{api.history().then(setItems).catch(e=>setError(e.message))},[]);return <div className="p-8"><h1 className="text-2xl font-bold">Report history</h1>{error&&<p role="alert">{error}</p>}{!items.length&&<p>No saved reports yet. <Link to="/upload">Start analysis</Link></p>}{items.map(item=><p className="p-3 border my-2" key={item.analysis_id}><Link to={`/report/${item.analysis_id}`}>{item.patient_id||item.analysis_id.slice(0,8)} · {new Date(item.created_at).toLocaleString()} · {item.signed?'Signed':'Draft'}</Link></p>)}</div>}
