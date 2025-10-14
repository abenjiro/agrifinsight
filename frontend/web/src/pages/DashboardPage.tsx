import { useState, useEffect } from 'react'
import {
  BarChart3,
  Camera,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Upload,
  ArrowUp,
  ArrowDown
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
      <div className="w-full flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-pink-600"></div>
      </div>
    )
  }

  return (
    <>
      {/* Stats Cards Row */}
      <div className="w-full max-w-full px-3 mb-6 sm:w-1/2 sm:flex-none xl:mb-0 xl:w-1/4">
        <div className="relative flex flex-col min-w-0 break-words bg-white shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="flex-auto p-4">
            <div className="flex flex-row -mx-3">
              <div className="flex-none w-2/3 max-w-full px-3">
                <div>
                  <p className="mb-0 font-sans font-semibold leading-normal text-sm">Total Farms</p>
                  <h5 className="mb-0 font-bold">
                    {farms.length}
                    <span className="leading-normal text-sm font-weight-bolder text-lime-500"> +12%</span>
                  </h5>
                </div>
              </div>
              <div className="px-3 text-right basis-1/3">
                <div className="inline-block w-12 h-12 text-center rounded-lg bg-gradient-to-tl from-purple-700 to-pink-500">
                  <BarChart3 className="w-6 h-6 text-white relative top-3.5 left-3" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full max-w-full px-3 mb-6 sm:w-1/2 sm:flex-none xl:mb-0 xl:w-1/4">
        <div className="relative flex flex-col min-w-0 break-words bg-white shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="flex-auto p-4">
            <div className="flex flex-row -mx-3">
              <div className="flex-none w-2/3 max-w-full px-3">
                <div>
                  <p className="mb-0 font-sans font-semibold leading-normal text-sm">Analysis Today</p>
                  <h5 className="mb-0 font-bold">
                    3
                    <span className="leading-normal text-sm font-weight-bolder text-lime-500"> +5%</span>
                  </h5>
                </div>
              </div>
              <div className="px-3 text-right basis-1/3">
                <div className="inline-block w-12 h-12 text-center rounded-lg bg-gradient-to-tl from-blue-600 to-cyan-400">
                  <Camera className="w-6 h-6 text-white relative top-3.5 left-3" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full max-w-full px-3 mb-6 sm:w-1/2 sm:flex-none xl:mb-0 xl:w-1/4">
        <div className="relative flex flex-col min-w-0 break-words bg-white shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="flex-auto p-4">
            <div className="flex flex-row -mx-3">
              <div className="flex-none w-2/3 max-w-full px-3">
                <div>
                  <p className="mb-0 font-sans font-semibold leading-normal text-sm">Issues Detected</p>
                  <h5 className="mb-0 font-bold">
                    1
                    <span className="leading-normal text-sm font-weight-bolder text-red-600"> -2%</span>
                  </h5>
                </div>
              </div>
              <div className="px-3 text-right basis-1/3">
                <div className="inline-block w-12 h-12 text-center rounded-lg bg-gradient-to-tl from-red-600 to-rose-400">
                  <AlertTriangle className="w-6 h-6 text-white relative top-3.5 left-3" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full max-w-full px-3 sm:w-1/2 sm:flex-none xl:w-1/4">
        <div className="relative flex flex-col min-w-0 break-words bg-white shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="flex-auto p-4">
            <div className="flex flex-row -mx-3">
              <div className="flex-none w-2/3 max-w-full px-3">
                <div>
                  <p className="mb-0 font-sans font-semibold leading-normal text-sm">Healthy Crops</p>
                  <h5 className="mb-0 font-bold">
                    85%
                    <span className="leading-normal text-sm font-weight-bolder text-lime-500"> +3%</span>
                  </h5>
                </div>
              </div>
              <div className="px-3 text-right basis-1/3">
                <div className="inline-block w-12 h-12 text-center rounded-lg bg-gradient-to-tl from-green-600 to-lime-400">
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
            <h6 className="mb-0 font-bold">Your Farms</h6>
          </div>
          <div className="flex-auto p-4">
            <div className="space-y-4">
              {farms.map((farm) => (
                <div
                  key={farm.id}
                  className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:shadow-soft-xs transition-all"
                >
                  <div className="flex items-center">
                    <div className="w-12 h-12 bg-gradient-to-tl from-purple-700 to-pink-500 rounded-lg flex items-center justify-center text-white font-bold mr-4">
                      {farm.name.charAt(0)}
                    </div>
                    <div>
                      <h6 className="mb-0 leading-normal text-sm font-semibold">{farm.name}</h6>
                      <p className="mb-0 leading-tight text-xs text-slate-400">
                        {farm.location} • {farm.size} acres • {farm.crop_type}
                      </p>
                    </div>
                  </div>
                  <span className="inline-block px-2.5 py-1 text-xs font-semibold text-center text-green-600 bg-green-200 rounded-lg">
                    Active
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Recent Analysis Card */}
      <div className="w-full max-w-full px-3 mt-6 md:w-5/12 md:flex-none">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
            <h6 className="mb-0 font-bold">Recent Analysis</h6>
          </div>
          <div className="flex-auto p-4">
            <div className="space-y-4">
              {recentAnalysis.map((analysis) => (
                <div key={analysis.id} className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-gradient-to-tl from-red-600 to-rose-400 rounded-lg flex items-center justify-center">
                      <AlertTriangle className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-gray-900">{analysis.disease_detected}</p>
                    <p className="text-xs text-gray-500">
                      Confidence: {Math.round(analysis.confidence * 100)}%
                    </p>
                  </div>
                  <span className="inline-block px-2.5 py-1 text-xs font-semibold text-red-600 bg-red-200 rounded-lg">
                    Disease
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions Card */}
      <div className="w-full max-w-full px-3 mt-6">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
            <h6 className="mb-0 font-bold">Quick Actions</h6>
          </div>
          <div className="flex-auto p-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button className="inline-block px-6 py-3 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-purple-700 to-pink-500 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md bg-150 bg-x-25 hover:scale-102 hover:shadow-soft-xs active:opacity-85">
                <Upload className="w-4 h-4 inline mr-2" />
                Upload Image
              </button>
              <button className="inline-block px-6 py-3 font-bold text-center text-slate-700 uppercase align-middle transition-all bg-transparent border border-solid rounded-lg cursor-pointer border-slate-700 leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md bg-150 bg-x-25 hover:scale-102 hover:shadow-soft-xs active:opacity-85">
                <BarChart3 className="w-4 h-4 inline mr-2" />
                View Reports
              </button>
              <button className="inline-block px-6 py-3 font-bold text-center text-slate-700 uppercase align-middle transition-all bg-transparent border border-solid rounded-lg cursor-pointer border-slate-700 leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md bg-150 bg-x-25 hover:scale-102 hover:shadow-soft-xs active:opacity-85">
                <TrendingUp className="w-4 h-4 inline mr-2" />
                Get Recommendations
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
