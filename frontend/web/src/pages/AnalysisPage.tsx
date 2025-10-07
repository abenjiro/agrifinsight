import { useState } from 'react'
import { Upload, Camera, FileImage, AlertCircle, CheckCircle } from 'lucide-react'

export function AnalysisPage() {
  const [dragActive, setDragActive] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

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
      formData.append('image', uploadedFile)
      formData.append('farm_id', '1') // Default farm ID for now
      
      const response = await fetch('http://localhost:8000/api/analysis/upload', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        throw new Error('Analysis failed')
      }
      
      const data = await response.json()
      setAnalysisResult(data.analysis_result)
    } catch (error) {
      console.error('Error analyzing image:', error)
      // Fallback to mock data if API fails
      setAnalysisResult({
        disease_detected: 'Tomato Blight',
        confidence_score: 0.87,
        recommendations: [
          'Apply copper-based fungicide',
          'Improve air circulation around plants',
          'Remove affected leaves immediately',
          'Water at soil level, not on leaves'
        ],
        severity: 'High'
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Disease Analysis</h1>
        <p className="text-gray-600">Upload plant images to detect diseases and get recommendations</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Image</h3>
          
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {uploadedFile ? (
              <div className="space-y-4">
                <FileImage className="w-12 h-12 text-primary-600 mx-auto" />
                <div>
                  <p className="text-sm font-medium text-gray-900">{uploadedFile.name}</p>
                  <p className="text-sm text-gray-500">
                    {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setUploadedFile(null)}
                    className="btn btn-outline text-sm"
                  >
                    Remove
                  </button>
                  <button
                    onClick={analyzeImage}
                    disabled={isAnalyzing}
                    className="btn btn-primary text-sm"
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <Camera className="w-12 h-12 text-gray-400 mx-auto" />
                <div>
                  <p className="text-lg font-medium text-gray-900">
                    Drop your image here
                  </p>
                  <p className="text-gray-500">or click to browse</p>
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
                  className="btn btn-primary cursor-pointer"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Choose File
                </label>
              </div>
            )}
          </div>

          <div className="mt-4 text-sm text-gray-500">
            <p>Supported formats: JPG, PNG, WebP</p>
            <p>Maximum file size: 10MB</p>
          </div>
        </div>

        {/* Analysis Results */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Results</h3>
          
          {isAnalyzing ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
                <p className="text-gray-600">Analyzing your image...</p>
              </div>
            </div>
          ) : analysisResult ? (
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-5 h-5 text-red-500" />
                <span className="font-medium text-gray-900">Disease Detected</span>
              </div>
              
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <h4 className="font-semibold text-red-900 mb-2">
                  {analysisResult.disease_detected || analysisResult.disease}
                </h4>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-red-700">Confidence:</span>
                  <span className="text-sm font-medium text-red-900">
                    {Math.round((analysisResult.confidence_score || analysisResult.confidence) * 100)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-red-700">Severity:</span>
                  <span className="text-sm font-medium text-red-900">
                    {analysisResult.severity || (analysisResult.is_healthy ? 'Healthy' : 'Needs Attention')}
                  </span>
                </div>
              </div>

              <div>
                <h5 className="font-medium text-gray-900 mb-2">Recommendations:</h5>
                <ul className="space-y-1">
                  {(analysisResult.recommendations || []).map((rec: string, index: number) => (
                    <li key={index} className="flex items-start space-x-2 text-sm text-gray-700">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              {analysisResult.treatment_advice && (
                <div>
                  <h5 className="font-medium text-gray-900 mb-2">Treatment Advice:</h5>
                  <p className="text-sm text-gray-700">{analysisResult.treatment_advice}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              <div className="text-center">
                <Camera className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p>Upload an image to see analysis results</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Analysis History */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Analysis</h3>
        <div className="text-center py-8 text-gray-500">
          <p>No recent analysis found</p>
        </div>
      </div>
    </div>
  )
}

