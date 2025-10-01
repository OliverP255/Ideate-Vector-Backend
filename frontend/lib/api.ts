import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// API methods
export const api = {
  // Health check
  async getHealth() {
    const response = await apiClient.get('/health')
    return response.data
  },

  // Map data
  async getMapData(bounds?: string, zoomLevel?: number, limit?: number) {
    const params = new URLSearchParams()
    if (bounds) params.append('bounds', bounds)
    if (zoomLevel) params.append('zoom_level', zoomLevel.toString())
    if (limit) params.append('limit', limit.toString())
    
    const response = await apiClient.get(`/map?${params.toString()}`)
    return response.data
  },

  // Click handling
  async handleClick(lat: number, lon: number, query?: string, userId?: string) {
    const response = await apiClient.post('/click', {
      lat,
      lon,
      query,
    }, {
      params: { user_id: userId }
    })
    return response.data
  },

  // User overlay
  async getUserOverlay(userId: string) {
    const response = await apiClient.get(`/user/${userId}/overlay`)
    return response.data
  },

  // Document ingestion
  async ingestDocument(filePath: string, fileType: string, metadata?: any) {
    const response = await apiClient.post('/ingest', {
      file_path: filePath,
      file_type: fileType,
      metadata,
    })
    return response.data
  },
}

export default api
