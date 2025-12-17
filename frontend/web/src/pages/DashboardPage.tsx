import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  Camera,
  AlertTriangle,
  CheckCircle,
  Upload,
  Sprout,
  MapPin,
  Plus,
  ArrowRight
} from 'lucide-react'
import { farmService } from '../services/api'
import api from '../services/api'

interface Farm {
  id: number
  name: string
  address?: string
  latitude?: number
  longitude?: number
  size?: number
  size_unit?: string
  soil_type?: string
  climate_zone?: string
  avg_temperature?: number
  country?: string
  region?: string
  created_at: string
}

interface DashboardStats {
  totalFarms: number
  totalSize: number
  analysisCount: number
  issuesDetected: number
  healthyPercentage: number
}

interface RecentAnalysis {
  id: number
  image_id: number
  disease_detected: string
  confidence_score: number
  severity: string
  created_at: string
  filename?: string
}

export function DashboardPage() {
  const [farms, setFarms] = useState<Farm[]>([])
  const [recentAnalysis, setRecentAnalysis] = useState<RecentAnalysis[]>([])
  const [stats, setStats] = useState<DashboardStats>({
    totalFarms: 0,
    totalSize: 0,
    analysisCount: 0,
    issuesDetected: 0,
    healthyPercentage: 0
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {

      // Fetch farms using farmService
      const response = await farmService.getFarms()
      const farmsData: any = response.data || response
      setFarms(farmsData)

      // Calculate farm stats
      const totalSize = farmsData.reduce((sum: number, farm: Farm) =>
        sum + (farm.size || 0), 0
      )

      // Fetch analysis history to calculate stats
      let analysisCount = 0
      let issuesDetected = 0
      let healthyCount = 0

      try {
        const analysisResponse = await api.get('/analysis/history')
        const responseData = analysisResponse.data

        // Handle different response formats
        let analysisData = []
        if (Array.isArray(responseData)) {
          analysisData = responseData
        } else if (responseData && Array.isArray(responseData.data)) {
          analysisData = responseData.data
        }

        analysisCount = analysisData.length

        // Set recent analysis (top 3)
        setRecentAnalysis(analysisData.slice(0, 3))

        // Calculate issues and healthy percentage
        analysisData.forEach((analysis: any) => {
          if (analysis.disease_detected && !analysis.disease_detected.toLowerCase().includes('healthy')) {
            issuesDetected++
          } else {
            healthyCount++
          }
        })
      } catch (error) {
        console.error('Error fetching analysis data:', error)
      }

      const healthyPercentage = analysisCount > 0
        ? Math.round((healthyCount / analysisCount) * 100)
        : 0

      setStats({
        totalFarms: farmsData.length,
        totalSize: Math.round(totalSize),
        analysisCount,
        issuesDetected,
        healthyPercentage
      })
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="w-full flex items-center justify-center py-20">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <>

      {/* Stats Cards Row */}
      <div className="w-full max-w-full px-3 mb-6 sm:w-1/2 sm:flex-none xl:mb-0 xl:w-1/4">
        <div className="relative flex flex-col min-w-0 break-words bg-white shadow-soft-xl rounded-2xl bg-clip-border hover:shadow-soft-2xl transition">
          <div className="flex-auto p-4">
            <div className="flex flex-row -mx-3">
              <div className="flex-none w-2/3 max-w-full px-3">
                <div>
                  <p className="mb-0 font-sans font-semibold leading-normal text-sm text-gray-600">Total Farms</p>
                  <h5 className="mb-0 font-bold text-2xl">
                    {stats.totalFarms}
                  </h5>
                  <p className="text-xs text-gray-500 mt-1">{stats.totalSize} acres total</p>
                </div>
              </div>
              <div className="px-3 text-right basis-1/3">
                <div className="inline-block w-12 h-12 text-center rounded-lg bg-gradient-to-tl from-green-600 to-lime-400 shadow-lg">
                  <Sprout className="w-6 h-6 text-white relative top-3.5 left-3" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full max-w-full px-3 mb-6 sm:w-1/2 sm:flex-none xl:mb-0 xl:w-1/4">
        <div className="relative flex flex-col min-w-0 break-words bg-white shadow-soft-xl rounded-2xl bg-clip-border hover:shadow-soft-2xl transition">
          <div className="flex-auto p-4">
            <div className="flex flex-row -mx-3">
              <div className="flex-none w-2/3 max-w-full px-3">
                <div>
                  <p className="mb-0 font-sans font-semibold leading-normal text-sm text-gray-600">Analysis Done</p>
                  <h5 className="mb-0 font-bold text-2xl">
                    {stats.analysisCount}
                  </h5>
                  <p className="text-xs text-gray-500 mt-1">All time</p>
                </div>
              </div>
              <div className="px-3 text-right basis-1/3">
                <div className="inline-block w-12 h-12 text-center rounded-lg bg-gradient-to-tl from-blue-600 to-cyan-400 shadow-lg">
                  <Camera className="w-6 h-6 text-white relative top-3.5 left-3" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full max-w-full px-3 mb-6 sm:w-1/2 sm:flex-none xl:mb-0 xl:w-1/4">
        <div className="relative flex flex-col min-w-0 break-words bg-white shadow-soft-xl rounded-2xl bg-clip-border hover:shadow-soft-2xl transition">
          <div className="flex-auto p-4">
            <div className="flex flex-row -mx-3">
              <div className="flex-none w-2/3 max-w-full px-3">
                <div>
                  <p className="mb-0 font-sans font-semibold leading-normal text-sm text-gray-600">Issues Detected</p>
                  <h5 className="mb-0 font-bold text-2xl">
                    {stats.issuesDetected}
                  </h5>
                  <p className={`text-xs mt-1 ${stats.issuesDetected > 0 ? 'text-orange-600' : 'text-green-600'}`}>
                    {stats.issuesDetected > 0 ? 'Needs attention' : 'No issues found'}
                  </p>
                </div>
              </div>
              <div className="px-3 text-right basis-1/3">
                <div className="inline-block w-12 h-12 text-center rounded-lg bg-gradient-to-tl from-amber-600 to-yellow-400 shadow-lg">
                  <AlertTriangle className="w-6 h-6 text-white relative top-3.5 left-3" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full max-w-full px-3 sm:w-1/2 sm:flex-none xl:w-1/4">
        <div className="relative flex flex-col min-w-0 break-words bg-white shadow-soft-xl rounded-2xl bg-clip-border hover:shadow-soft-2xl transition">
          <div className="flex-auto p-4">
            <div className="flex flex-row -mx-3">
              <div className="flex-none w-2/3 max-w-full px-3">
                <div>
                  <p className="mb-0 font-sans font-semibold leading-normal text-sm text-gray-600">Farm Health</p>
                  <h5 className="mb-0 font-bold text-2xl">
                    {stats.healthyPercentage}%
                  </h5>
                  <p className={`text-xs mt-1 ${
                    stats.healthyPercentage >= 80 ? 'text-green-600' :
                    stats.healthyPercentage >= 60 ? 'text-yellow-600' :
                    stats.healthyPercentage >= 40 ? 'text-orange-600' :
                    'text-red-600'
                  }`}>
                    {stats.healthyPercentage >= 80 ? 'Excellent status' :
                     stats.healthyPercentage >= 60 ? 'Good status' :
                     stats.healthyPercentage >= 40 ? 'Fair status' :
                     stats.analysisCount > 0 ? 'Needs attention' : 'No data yet'}
                  </p>
                </div>
              </div>
              <div className="px-3 text-right basis-1/3">
                <div className="inline-block w-12 h-12 text-center rounded-lg bg-gradient-to-tl from-emerald-600 to-teal-400 shadow-lg">
                  <CheckCircle className="w-6 h-6 text-white relative top-3.5 left-3" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Your Farms Card */}
      <div className="w-full max-w-full px-3 mt-6 md:w-7/12 md:flex-none">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
            <div className="flex items-center justify-between">
              <h6 className="mb-0 font-bold">Your Farms</h6>
              <Link
                to="/farms"
                className="text-xs font-semibold text-green-600 hover:text-green-700 transition"
              >
                View All
              </Link>
            </div>
          </div>
          <div className="flex-auto p-4">
            {farms.length > 0 ? (
              <div className="space-y-4">
                {farms.slice(0, 5).map((farm) => (
                  <div
                    key={farm.id}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:shadow-soft-xs transition-all cursor-pointer"
                  >
                    <div className="flex items-center flex-1">
                      <div className="w-12 h-12 bg-gradient-to-tl from-purple-700 to-pink-500 rounded-lg flex items-center justify-center text-white font-bold mr-4">
                        {farm.name.charAt(0).toUpperCase()}
                      </div>
                      <div className="flex-1">
                        <h6 className="mb-0 leading-normal text-sm font-semibold">{farm.name}</h6>
                        <p className="mb-0 leading-tight text-xs text-slate-400">
                          <MapPin className="w-3 h-3 inline mr-1" />
                          {farm.address || farm.region || farm.country || 'Location not set'}
                          {farm.size && (
                            <>
                              {' '} • {farm.size} {farm.size_unit || 'acres'}
                            </>
                          )}
                          {farm.soil_type && (
                            <>
                              {' '} • {farm.soil_type}
                            </>
                          )}
                        </p>
                      </div>
                    </div>
                    <span className="inline-block px-2.5 py-1 text-xs font-semibold text-center text-green-600 bg-green-200 rounded-lg">
                      Active
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="inline-block w-16 h-16 text-center rounded-xl bg-gray-100 mb-4">
                  <Sprout className="w-8 h-8 text-gray-300 relative top-4 left-4" />
                </div>
                <p className="text-sm font-medium text-gray-500 mb-2">No farms yet</p>
                <p className="text-xs text-gray-400 mb-4">Add your first farm to get started</p>
                <Link
                  to="/farms"
                  className="inline-block px-6 py-2.5 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-green-600 to-lime-400 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85"
                >
                  <Plus className="w-3 h-3 inline mr-2" />
                  Add Farm
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recent Analysis Card */}
      <div className="w-full max-w-full px-3 mt-6 md:w-5/12 md:flex-none">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
            <div className="flex items-center justify-between">
              <h6 className="mb-0 font-bold">Recent Analysis</h6>
              <Link
                to="/analysis"
                className="text-xs font-semibold text-green-600 hover:text-green-700 transition"
              >
                View All
              </Link>
            </div>
          </div>
          <div className="flex-auto p-4">
            {recentAnalysis.length > 0 ? (
              <div className="space-y-3">
                {recentAnalysis.map((analysis) => {
                  const isHealthy = analysis.disease_detected?.toLowerCase().includes('healthy')
                  const severityColor =
                    analysis.severity === 'High' || analysis.severity === 'Severe' ? 'text-red-600 bg-red-100' :
                    analysis.severity === 'Moderate' || analysis.severity === 'Medium' ? 'text-orange-600 bg-orange-100' :
                    analysis.severity === 'Low' || analysis.severity === 'Mild' ? 'text-yellow-600 bg-yellow-100' :
                    'text-green-600 bg-green-100'

                  return (
                    <div
                      key={analysis.id}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:shadow-soft-xs transition-all"
                    >
                      <div className="flex items-center flex-1">
                        <div className={`w-10 h-10 ${isHealthy ? 'bg-gradient-to-tl from-green-600 to-lime-400' : 'bg-gradient-to-tl from-red-600 to-orange-400'} rounded-lg flex items-center justify-center mr-3`}>
                          {isHealthy ? (
                            <CheckCircle className="w-5 h-5 text-white" />
                          ) : (
                            <AlertTriangle className="w-5 h-5 text-white" />
                          )}
                        </div>
                        <div className="flex-1">
                          <h6 className="mb-0 leading-normal text-xs font-semibold">
                            {analysis.disease_detected || 'Unknown'}
                          </h6>
                          <p className="mb-0 leading-tight text-xs text-slate-400">
                            {analysis.filename || `Analysis #${analysis.id}`} • {Math.round(analysis.confidence_score * 100)}% confidence
                          </p>
                        </div>
                      </div>
                      <span className={`inline-block px-2 py-1 text-xs font-semibold text-center rounded-lg ${severityColor}`}>
                        {analysis.severity || 'N/A'}
                      </span>
                    </div>
                  )
                })}
                <Link
                  to="/analysis"
                  className="block text-center text-xs font-semibold text-blue-600 hover:text-blue-700 transition pt-2"
                >
                  View All Analysis <ArrowRight className="w-3 h-3 inline ml-1" />
                </Link>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="inline-block w-16 h-16 text-center rounded-xl bg-gray-100 mb-4">
                  <Camera className="w-8 h-8 text-gray-300 relative top-4 left-4" />
                </div>
                <p className="text-sm font-medium text-gray-500 mb-2">No analysis yet</p>
                <p className="text-xs text-gray-400 mb-4">Upload crop images to get AI-powered insights</p>
                <Link
                  to="/analysis"
                  className="inline-block px-6 py-2.5 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-blue-600 to-cyan-400 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85"
                >
                  <Upload className="w-3 h-3 inline mr-2" />
                  Start Analysis
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions Card */}
      {/* <div className="w-full max-w-full px-3 mt-6">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
            <h6 className="mb-0 font-bold">Quick Actions</h6>
          </div>
          <div className="flex-auto p-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Link
                to="/analysis"
                className="inline-block px-6 py-3 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-purple-700 to-pink-500 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md bg-150 bg-x-25 hover:scale-102 hover:shadow-soft-xs active:opacity-85"
              >
                <Upload className="w-4 h-4 inline mr-2" />
                Upload Image
              </Link>
              <Link
                to="/farms"
                className="inline-block px-6 py-3 font-bold text-center text-slate-700 uppercase align-middle transition-all bg-transparent border border-solid rounded-lg cursor-pointer border-slate-700 leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md bg-150 bg-x-25 hover:scale-102 hover:shadow-soft-xs active:opacity-85"
              >
                <Sprout className="w-4 h-4 inline mr-2" />
                Manage Farms
              </Link>
              <Link
                to="/ai-features"
                className="inline-block px-6 py-3 font-bold text-center text-slate-700 uppercase align-middle transition-all bg-transparent border border-solid rounded-lg cursor-pointer border-slate-700 leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md bg-150 bg-x-25 hover:scale-102 hover:shadow-soft-xs active:opacity-85"
              >
                <TrendingUp className="w-4 h-4 inline mr-2" />
                Explore AI Features
              </Link>
            </div>
          </div>
        </div>
      </div> */}
    </>
  )
}
