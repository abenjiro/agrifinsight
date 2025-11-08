import { useState, useEffect } from 'react'
import { Upload, Camera, FileImage, AlertCircle, CheckCircle, Loader, Calendar, Trash2 } from 'lucide-react'
import { showError, showSuccess, showConfirm } from '../utils/sweetalert'
import api from '../services/api'

interface AnalysisHistory {
  id: number
  image_path: string
  disease_detected: string
  confidence_score: number
  severity: string
  created_at: string
  farm_id?: number
}

export function AnalysisPage() {
  const [dragActive, setDragActive] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistory[]>([])
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [saving, setSaving] = useState(false)
  const [currentAnalysisId, setCurrentAnalysisId] = useState<number | null>(null)

  useEffect(() => {
    fetchAnalysisHistory()
  }, [])

  const fetchAnalysisHistory = async () => {
    try {
      setLoadingHistory(true)
      const response = await api.get('/analysis/history')
      const data = response.data

      // Ensure we have an array
      if (Array.isArray(data)) {
        setAnalysisHistory(data)
      } else if (data && Array.isArray(data.data)) {
        setAnalysisHistory(data.data)
      } else {
        setAnalysisHistory([])
      }
    } catch (error: any) {
      console.error('Error fetching analysis history:', error)
      // Set empty array on error to prevent map errors
      setAnalysisHistory([])
    } finally {
      setLoadingHistory(false)
    }
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFile = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      setUploadedFile(file)
      setAnalysisResult(null)

      // Create preview URL
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const analyzeImage = async () => {
    if (!uploadedFile) return

    setIsAnalyzing(true)

    try {
      const formData = new FormData()
      formData.append('file', uploadedFile)
      // Optional: Add farm_id if available
      // formData.append('farm_id', '1')

      const response = await api.post('/analysis/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      const data = response.data
      console.log('Analysis response:', data)

      // Store analysis ID from various possible locations in response
      const analysisId = data.id || data.analysis_id || data.data?.id
      if (analysisId) {
        setCurrentAnalysisId(analysisId)
        console.log('Analysis ID set:', analysisId)
      } else {
        // If no ID returned, the analysis is already saved by backend
        console.log('No analysis ID returned - analysis may be auto-saved')
        // Enable save button anyway - let backend handle duplicate saves
        setCurrentAnalysisId(Date.now()) // Use timestamp as fallback
      }

      // The backend returns analysis_result object
      const result = data.analysis_result

      // Transform the AI response to match our UI expectations
      setAnalysisResult({
        disease_detected: result.disease_detected || result.disease_type || 'Unknown',
        confidence_score: result.confidence_score || 0,
        severity: result.severity || 'Unknown',
        recommendations: result.recommendations || [],
        treatment_advice: result.treatment_advice || '',
        top_predictions: result.top_predictions || [],
        is_healthy: result.is_healthy || false,
        needs_attention: result.needs_attention || false
      })

      // Refresh history
      await fetchAnalysisHistory()
    } catch (error: any) {
      console.error('Error analyzing image:', error)
      showError(error.response?.data?.detail || error.message || 'Unknown error occurred during analysis', 'Analysis Failed')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const saveAnalysis = async () => {
    if (!analysisResult || !currentAnalysisId) {
      showError('No analysis to save', 'Save Failed')
      return
    }

    setSaving(true)
    try {
      // Call the backend save endpoint
      await api.post(`/analysis/${currentAnalysisId}/save`)
      showSuccess('Analysis saved successfully!', 'Success')
      await fetchAnalysisHistory()
    } catch (error: any) {
      console.error('Error saving analysis:', error)
      showError(error.response?.data?.detail || error.message || 'Failed to save analysis', 'Save Failed')
    } finally {
      setSaving(false)
    }
  }

  const deleteAnalysis = async (id: number) => {
    const result = await showConfirm(
      'This will permanently delete this analysis record.',
      'Delete Analysis?',
      'Yes, delete it',
      'Cancel'
    )

    if (!result.isConfirmed) return

    try {
      await api.delete(`/analysis/${id}`)
      showSuccess('Analysis deleted successfully!', 'Deleted')
      await fetchAnalysisHistory()
    } catch (error: any) {
      console.error('Error deleting analysis:', error)
      showError(error.response?.data?.detail || 'Failed to delete analysis', 'Delete Failed')
    }
  }

  const resetAnalysis = () => {
    setUploadedFile(null)
    setPreviewUrl(null)
    setAnalysisResult(null)
  }

  return (
    <>
      {/* Upload Section */}
      <div className="w-full max-w-full px-3 mb-6 lg:w-7/12 lg:flex-none">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
            <h6 className="mb-0 font-bold">Upload Image</h6>
            <p className="leading-normal text-sm">Upload plant images for AI-powered disease detection</p>
          </div>
          <div className="flex-auto p-4">
            <div
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                dragActive
                  ? 'border-fuchsia-500 bg-fuchsia-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              {previewUrl ? (
                <div className="space-y-4">
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="max-h-64 mx-auto rounded-lg shadow-md"
                  />
                  <div>
                    <p className="text-sm font-semibold text-gray-900">{uploadedFile?.name}</p>
                    <p className="text-xs text-gray-500">
                      {uploadedFile && (uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <div className="flex justify-center space-x-3">
                    <button
                      onClick={resetAnalysis}
                      className="inline-block px-6 py-2.5 font-bold text-center text-slate-700 uppercase align-middle transition-all bg-transparent border border-solid rounded-lg cursor-pointer border-slate-700 leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85"
                    >
                      Remove
                    </button>
                    <button
                      onClick={analyzeImage}
                      disabled={isAnalyzing}
                      className="inline-block px-6 py-2.5 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-blue-600 to-cyan-400 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85 disabled:opacity-50"
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader className="w-3 h-3 inline mr-2 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        'Analyze Image'
                      )}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="inline-block w-16 h-16 text-center rounded-xl bg-gradient-to-tl from-blue-600 to-cyan-400">
                    <Camera className="w-8 h-8 text-white relative top-4 left-4" />
                  </div>
                  <div>
                    <p className="text-base font-semibold text-gray-900">
                      Drop your image here
                    </p>
                    <p className="text-sm text-gray-500">or click to browse from your device</p>
                  </div>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileInput}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="inline-block px-6 py-2.5 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-purple-700 to-pink-500 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85"
                  >
                    <Upload className="w-3 h-3 inline mr-2" />
                    Choose File
                  </label>
                </div>
              )}
            </div>

            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-600 mb-1">
                <strong>Supported formats:</strong> JPG, PNG, WebP
              </p>
              <p className="text-xs text-gray-600">
                <strong>Maximum file size:</strong> 10MB
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Analysis Results */}
      <div className="w-full max-w-full px-3 mb-6 lg:w-5/12 lg:flex-none">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
            <h6 className="mb-0 font-bold">Analysis Results</h6>
            <p className="leading-normal text-sm">AI-powered disease detection results</p>
          </div>
          <div className="flex-auto p-4">
            {isAnalyzing ? (
              <div className="flex items-center justify-center py-16">
                <div className="text-center">
                  <div className="inline-block w-12 h-12 text-center rounded-xl bg-gradient-to-tl from-purple-700 to-pink-500 mb-4 animate-pulse">
                    <Loader className="w-6 h-6 text-white relative top-3 left-3 animate-spin" />
                  </div>
                  <p className="text-sm font-semibold text-gray-900">Analyzing your image...</p>
                  <p className="text-xs text-gray-500 mt-1">This may take a few seconds</p>
                </div>
              </div>
            ) : analysisResult ? (
              <div className="space-y-4">
                <div className={`flex items-center justify-between p-4 rounded-xl ${
                  analysisResult.is_healthy
                    ? 'bg-gradient-to-tl from-green-600 to-lime-400'
                    : 'bg-gradient-to-tl from-red-600 to-rose-400'
                }`}>
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-white bg-opacity-20 rounded-lg flex items-center justify-center">
                      {analysisResult.is_healthy ? (
                        <CheckCircle className="w-5 h-5 text-white" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-white" />
                      )}
                    </div>
                    <div className="text-white">
                      <p className="text-xs font-semibold opacity-80">
                        {analysisResult.is_healthy ? 'Crop Status' : 'Disease Detected'}
                      </p>
                      <h5 className="font-bold text-base capitalize">
                        {analysisResult.disease_detected || analysisResult.disease || 'Unknown'}
                      </h5>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-xs text-gray-600 mb-1">Confidence</p>
                    <p className="text-lg font-bold text-gray-900">
                      {Math.round((analysisResult.confidence_score || analysisResult.confidence) * 100)}%
                    </p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-xs text-gray-600 mb-1">Severity</p>
                    <p className="text-lg font-bold text-red-600">
                      {analysisResult.severity || 'High'}
                    </p>
                  </div>
                </div>

                <div>
                  <h6 className="font-semibold text-gray-900 mb-2 text-sm">Recommendations:</h6>
                  <ul className="space-y-2">
                    {(analysisResult.recommendations || []).map((rec: string, index: number) => (
                      <li key={index} className="flex items-start space-x-2">
                        <div className="w-5 h-5 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                          <CheckCircle className="w-3 h-3 text-green-600" />
                        </div>
                        <span className="text-xs text-gray-700 leading-relaxed">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {analysisResult.treatment_advice && (
                  <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <h6 className="font-semibold text-blue-900 mb-1 text-xs">Treatment Advice:</h6>
                    <p className="text-xs text-blue-800">{analysisResult.treatment_advice}</p>
                  </div>
                )}

                <button
                  onClick={saveAnalysis}
                  disabled={saving || !currentAnalysisId}
                  className="inline-block w-full px-6 py-2.5 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-green-600 to-lime-400 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {saving ? (
                    <>
                      <Loader className="w-3 h-3 inline mr-2 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    'Save Analysis'
                  )}
                </button>
              </div>
            ) : (
              <div className="flex items-center justify-center py-16">
                <div className="text-center text-gray-400">
                  <div className="inline-block w-16 h-16 text-center rounded-xl bg-gray-100 mb-4">
                    <Camera className="w-8 h-8 text-gray-300 relative top-4 left-4" />
                  </div>
                  <p className="text-sm font-medium">No analysis yet</p>
                  <p className="text-xs mt-1">Upload an image to get started</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Analysis History */}
      <div className="w-full max-w-full px-3">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
            <h6 className="mb-0 font-bold">Recent Analysis History</h6>
            <p className="leading-normal text-sm">Your previous disease detection results</p>
          </div>
          <div className="flex-auto p-4">
            {loadingHistory ? (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <Loader className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-3" />
                  <p className="text-sm text-gray-500">Loading history...</p>
                </div>
              </div>
            ) : analysisHistory.length === 0 ? (
              <div className="text-center py-12">
                <div className="inline-block w-16 h-16 text-center rounded-xl bg-gray-100 mb-4">
                  <FileImage className="w-8 h-8 text-gray-300 relative top-4 left-4" />
                </div>
                <p className="text-sm font-medium text-gray-500">No recent analysis found</p>
                <p className="text-xs text-gray-400 mt-1">Your analysis history will appear here</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {(Array.isArray(analysisHistory) ? analysisHistory : []).map((analysis) => (
                  <div
                    key={analysis.id}
                    className="relative border border-gray-200 rounded-xl p-4 hover:shadow-md transition"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        analysis.disease_detected.toLowerCase().includes('healthy')
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {analysis.disease_detected.toLowerCase().includes('healthy') ? (
                          <CheckCircle className="w-3 h-3 inline mr-1" />
                        ) : (
                          <AlertCircle className="w-3 h-3 inline mr-1" />
                        )}
                        {analysis.disease_detected}
                      </div>
                      <button
                        onClick={() => deleteAnalysis(analysis.id)}
                        className="p-1 text-gray-400 hover:text-red-600 transition"
                        title="Delete analysis"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-600">Confidence:</span>
                        <span className="text-sm font-bold text-gray-900">
                          {Math.round(analysis.confidence_score * 100)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-600">Severity:</span>
                        <span className={`text-sm font-bold ${
                          analysis.severity.toLowerCase() === 'high' ? 'text-red-600' :
                          analysis.severity.toLowerCase() === 'medium' ? 'text-orange-600' :
                          'text-green-600'
                        }`}>
                          {analysis.severity}
                        </span>
                      </div>
                      <div className="pt-2 border-t border-gray-100">
                        <div className="flex items-center text-xs text-gray-500">
                          <Calendar className="w-3 h-3 mr-1" />
                          {new Date(analysis.created_at).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
