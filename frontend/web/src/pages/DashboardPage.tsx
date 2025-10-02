import { useState, useEffect } from 'react'
import { 
  BarChart3, 
  Camera, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle,
  Upload
} from 'lucide-react'
import type { Farm, AnalysisResult } from '../types'

export function DashboardPage() {
  const [farms, setFarms] = useState<Farm[]>([])
  const [recentAnalysis, setRecentAnalysis] = useState<AnalysisResult[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      setFarms([
        {
          id: '1',
          name: 'North Field',
          location: 'Iowa, USA',
          size: 50,
          crop_type: 'Corn',
          soil_type: 'Loam',
          user_id: '1',
          created_at: '2024-01-01',
          updated_at: '2024-01-01'
        },
        {
          id: '2',
          name: 'South Field',
          location: 'Iowa, USA',
          size: 30,
          crop_type: 'Soybean',
          soil_type: 'Clay',
          user_id: '1',
          created_at: '2024-01-01',
          updated_at: '2024-01-01'
        }
      ])
      setRecentAnalysis([
        {
          id: '1',
          farm_id: '1',
          image_url: '/api/placeholder/300/200',
          disease_detected: 'Corn Rust',
          confidence: 0.85,
          recommendations: ['Apply fungicide', 'Improve air circulation'],
          created_at: '2024-01-15'
        }
      ])
      setLoading(false)
    }, 1000)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600">Overview of your farms and recent analysis</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <BarChart3 className="h-8 w-8 text-primary-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Total Farms</p>
              <p className="text-2xl font-semibold text-gray-900">{farms.length}</p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Camera className="h-8 w-8 text-secondary-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Analysis Today</p>
              <p className="text-2xl font-semibold text-gray-900">3</p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <AlertTriangle className="h-8 w-8 text-red-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Issues Detected</p>
              <p className="text-2xl font-semibold text-gray-900">1</p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <CheckCircle className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Healthy Crops</p>
              <p className="text-2xl font-semibold text-gray-900">85%</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Farms Overview */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Your Farms</h3>
          <div className="space-y-4">
            {farms.map((farm) => (
              <div key={farm.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <h4 className="font-medium text-gray-900">{farm.name}</h4>
                  <p className="text-sm text-gray-600">{farm.location} • {farm.size} acres</p>
                  <p className="text-sm text-gray-500">{farm.crop_type}</p>
                </div>
                <div className="text-right">
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    Active
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Analysis */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Analysis</h3>
          <div className="space-y-4">
            {recentAnalysis.map((analysis) => (
              <div key={analysis.id} className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
                <div className="flex-shrink-0">
                  <img
                    src={analysis.image_url}
                    alt="Analysis"
                    className="h-12 w-12 rounded-lg object-cover"
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900">
                    {analysis.disease_detected}
                  </p>
                  <p className="text-sm text-gray-500">
                    Confidence: {Math.round(analysis.confidence * 100)}%
                  </p>
                </div>
                <div className="flex-shrink-0">
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                    Disease
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="btn btn-primary p-4 rounded-lg flex items-center justify-center space-x-2">
            <Upload className="w-5 h-5" />
            <span>Upload Image</span>
          </button>
          <button className="btn btn-outline p-4 rounded-lg flex items-center justify-center space-x-2">
            <BarChart3 className="w-5 h-5" />
            <span>View Reports</span>
          </button>
          <button className="btn btn-outline p-4 rounded-lg flex items-center justify-center space-x-2">
            <TrendingUp className="w-5 h-5" />
            <span>Get Recommendations</span>
          </button>
        </div>
      </div>
    </div>
  )
}

