const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
async function request(path, options = {}) {
  const key = localStorage.getItem('chestxpert_api_key')
  const res = await fetch(`${BASE_URL}/api/v1${path}`, {...options, headers: {...(key ? {'X-API-Key': key} : {}), ...options.headers}})
  if (!res.ok) {const error = await res.json().catch(() => ({})); throw new Error(typeof error.detail === 'string' ? error.detail : `Request failed (${res.status})`)}
  return res.json()
}
const json = (method, data) => ({method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)})
export const api = {
  health: () => request('/health'),
  identity: () => request('/identity'),
  uploadAnalysis: body => request('/analyses', {method:'POST', body}),
  getAnalysis: id => request(`/analyses/${id}`),
  history: () => request('/reviews'),
  review: id => request(`/reviews/${id}`),
  save: (id, data) => request(`/reviews/${id}`, json('PUT', data)),
  sign: (id, data) => request(`/reviews/${id}/sign`, json('POST', data)),
  templates: () => request('/templates'),
  saveTemplate: (id, data) => request(`/templates/${id}`, json('PUT', data)),
  uploadTemplate: file => { const body = new FormData(); body.append('file',file); return request('/templates/upload',{method:'POST',body}) },
  async image(id,kind) {
    const key = localStorage.getItem('chestxpert_api_key')
    const res = await fetch(`${BASE_URL}/api/v1/reviews/${id}/image/${kind}`,{headers:key?{'X-API-Key':key}:{}})
    if(!res.ok)return null
    return URL.createObjectURL(await res.blob())
  },
  async downloadReport(id, format) {
    const key = localStorage.getItem('chestxpert_api_key')
    const res = await fetch(`${BASE_URL}/api/v1/report/${id}/download?format=${format}`, {headers: key ? {'X-API-Key':key} : {}})
    if (!res.ok) throw new Error('Report export failed')
    return res.blob()
  },
}
