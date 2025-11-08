import { Link } from 'react-router-dom'
import {
  Brain,
  Target,
  TrendingUp,
  Satellite,
  AlertTriangle,
  CloudRain,
  Leaf,
  CheckCircle2,
  ArrowRight,
  Zap,
  Shield,
  BarChart3
} from 'lucide-react'
import homeImage from '../assets/img/home1.png'

const features = [
  {
    icon: Brain,
    title: 'AI Crop Disease Detection',
    description: 'Upload a photo of your crop and get instant disease identification powered by advanced deep learning models.',
    benefits: [
      'High accuracy disease detection',
      'Identify multiple crop diseases',
      'Get treatment recommendations',
      'Track disease progression over time'
    ],
    gradient: 'from-red-500 to-orange-500',
  },
  {
    icon: Target,
    title: 'Smart Planting Recommendations',
    description: 'AI analyzes your soil, climate, and market trends to recommend the best crops and optimal planting times.',
    benefits: [
      'Maximize yield potential',
      'Reduce crop failure risk',
      'Weather-based planting calendar',
      'Market price predictions'
    ],
    gradient: 'from-green-500 to-emerald-500',
  },
  {
    icon: TrendingUp,
    title: 'Harvest Yield Predictions',
    description: 'Machine learning models forecast your harvest yield and quality weeks in advance for better planning.',
    benefits: [
      'Accurate yield predictions',
      'Plan harvest logistics in advance',
      'Optimize resource allocation',
      'Maximize profit margins'
    ],
    gradient: 'from-blue-500 to-cyan-500',
  },
  {
    icon: Satellite,
    title: 'Satellite Crop Monitoring',
    description: 'Track crop health from space using NDVI analysis and satellite imagery for large-scale farm monitoring.',
    benefits: [
      'Monitor farms from anywhere',
      'Detect stress before visible',
      'Track growth patterns',
      'Historical land use insights'
    ],
    gradient: 'from-purple-500 to-pink-500',
  },
  {
    icon: CloudRain,
    title: 'Weather Intelligence',
    description: 'AI-powered weather forecasting and alerts help you make informed decisions about irrigation and protection.',
    benefits: [
      'Hyperlocal forecasts',
      'Rainfall predictions',
      'Frost and heat alerts',
      'Irrigation scheduling'
    ],
    gradient: 'from-indigo-500 to-blue-500',
  },
  {
    icon: AlertTriangle,
    title: 'Pest & Disease Alerts',
    description: 'Early warning system uses AI to predict pest outbreaks and disease spread in your region.',
    benefits: [
      'Prevent outbreaks proactively',
      'Region-based risk alerts',
      'Treatment timing optimization',
      'Reduce pesticide usage'
    ],
    gradient: 'from-yellow-500 to-amber-500',
  }
]

const stats = [
  { value: '98%', label: 'Disease Detection Accuracy', icon: Target },
  { value: 'Global', label: 'Coverage', icon: TrendingUp },
  { value: '1000+', label: 'Farms Monitored', icon: Leaf },
  { value: '24/7', label: 'AI Monitoring', icon: Zap }
]

export function AIFeaturesPage() {
  return (
    <div className="w-full min-h-screen bg-cover bg-no-repeat relative" style={{ backgroundImage: `url(${homeImage})` }}>
      {/* Animated Background Overlay - consistent with HomePage and AboutUs */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <div className="absolute top-20 left-10 w-96 h-96 bg-green-400/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-40 right-20 w-[500px] h-[500px] bg-blue-400/15 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute bottom-20 left-1/3 w-80 h-80 bg-purple-400/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Hero Section */}
      <div className="relative pt-24 pb-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8 md:p-12">
            <div className="flex flex-col lg:flex-row items-center gap-10">
              <div className="flex-1">
                <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4 leading-tight">
                  Transform Your Farm with
                  <span className="block mt-2 text-green-600"> Artificial Intelligence</span>
                </h1>

                <p className="text-lg text-gray-600 mb-8 leading-relaxed">
                  Harness the power of AI and satellite technology to maximize yields,
                  prevent crop diseases, and make data-driven decisions.
                </p>

                <div className="flex flex-wrap gap-4">
                  <Link
                    to="/register"
                    className="inline-flex items-center px-8 py-3.5 font-medium text-white bg-green-600 hover:bg-green-700 rounded-lg transition-colors shadow-md"
                  >
                    Get Started
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Link>

                  <Link
                    to="/analysis"
                    className="inline-flex items-center px-8 py-3.5 font-medium text-gray-700 bg-white border-2 border-gray-200 hover:border-green-600 hover:text-green-600 rounded-lg transition-colors"
                  >
                    Try Demo
                  </Link>
                </div>
              </div>

              {/* Hero Visual - Simplified */}
              <div className="flex-1">
                <div className="bg-gradient-to-br from-green-50 to-blue-50 rounded-2xl p-6 border border-gray-200">
                  <div className="text-7xl mb-4 text-center">🤖🌾</div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white rounded-lg p-3 shadow-sm">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-green-600" />
                        <span className="text-xs font-semibold">Disease Detection</span>
                      </div>
                      <p className="text-xs text-gray-600 mt-1">High accuracy</p>
                    </div>
                    <div className="bg-white rounded-lg p-3 shadow-sm">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-blue-600" />
                        <span className="text-xs font-semibold">Yield Optimization</span>
                      </div>
                      <p className="text-xs text-gray-600 mt-1">Data-driven</p>
                    </div>
                    <div className="bg-white rounded-lg p-3 shadow-sm">
                      <div className="flex items-center gap-2">
                        <Satellite className="w-4 h-4 text-purple-600" />
                        <span className="text-xs font-semibold">Satellite Monitoring</span>
                      </div>
                      <p className="text-xs text-gray-600 mt-1">Real-time</p>
                    </div>
                    <div className="bg-white rounded-lg p-3 shadow-sm">
                      <div className="flex items-center gap-2">
                        <CloudRain className="w-4 h-4 text-indigo-600" />
                        <span className="text-xs font-semibold">Weather Intelligence</span>
                      </div>
                      <p className="text-xs text-gray-600 mt-1">Forecasts</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="relative max-w-7xl mx-auto px-6 mb-16">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map((stat, index) => {
            const Icon = stat.icon
            return (
              <div
                key={index}
                className="bg-white border border-gray-200 rounded-xl p-6 text-center hover:border-green-300 hover:shadow-md transition-all"
              >
                <div className="inline-flex items-center justify-center w-12 h-12 mb-3 bg-green-100 rounded-lg">
                  <Icon className="w-6 h-6 text-green-600" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900">{stat.value}</h3>
                <p className="text-sm text-gray-600 mt-1">{stat.label}</p>
              </div>
            )
          })}
        </div>
      </div>

      {/* Features Grid */}
      <div className="relative max-w-7xl mx-auto px-6 mb-16">
        <div className="text-center mb-10">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
            AI-Powered Features
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Everything you need to optimize your farm operations
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <div
                key={index}
                className="bg-white border border-gray-200 rounded-xl p-6 hover:border-green-300 hover:shadow-md transition-all"
              >
                {/* Icon & Title */}
                <div className="flex items-start gap-4 mb-4">
                  <div className={`flex-shrink-0 w-12 h-12 flex items-center justify-center rounded-lg bg-gradient-to-br ${feature.gradient}`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900 mb-1">{feature.title}</h3>
                    <p className="text-sm text-gray-600">{feature.description}</p>
                  </div>
                </div>

                {/* Benefits List */}
                <div className="space-y-2 mt-4">
                  {feature.benefits.map((benefit, idx) => (
                    <div key={idx} className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 flex-shrink-0 mt-0.5 text-green-600" />
                      <span className="text-sm text-gray-700">{benefit}</span>
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Trust Section */}
      <div className="relative max-w-7xl mx-auto px-6 mb-16">
        <div className="bg-gray-900 rounded-2xl p-10 text-center">
          <h3 className="text-2xl font-bold text-white mb-8">Why Farmers Trust Our AI</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="flex flex-col items-center">
              <div className="w-14 h-14 flex items-center justify-center bg-green-100 rounded-lg mb-3">
                <Shield className="w-7 h-7 text-green-600" />
              </div>
              <h4 className="text-white font-semibold mb-2">Data Privacy</h4>
              <p className="text-gray-400 text-sm">Your farm data is encrypted and secure</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="w-14 h-14 flex items-center justify-center bg-blue-100 rounded-lg mb-3">
                <Brain className="w-7 h-7 text-blue-600" />
              </div>
              <h4 className="text-white font-semibold mb-2">Proven AI Models</h4>
              <p className="text-gray-400 text-sm">Trained on millions of crop images</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="w-14 h-14 flex items-center justify-center bg-purple-100 rounded-lg mb-3">
                <BarChart3 className="w-7 h-7 text-purple-600" />
              </div>
              <h4 className="text-white font-semibold mb-2">Real Results</h4>
              <p className="text-gray-400 text-sm">Measurable yield improvements</p>
            </div>
          </div>
        </div>
      </div>

      {/* Final CTA */}
      <div className="relative max-w-7xl mx-auto px-6 mb-12 pb-12">
        <div className="bg-green-600 rounded-2xl p-12 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Ready to Transform Your Farm?
          </h2>
          <p className="text-lg text-green-50 mb-8 max-w-2xl mx-auto">
            Join farmers who are using AI to increase yields and reduce costs
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link
              to="/register"
              className="inline-flex items-center px-8 py-3.5 font-medium text-green-600 bg-white hover:bg-green-50 rounded-lg transition-colors shadow-lg"
            >
              Get Started
              <ArrowRight className="w-4 h-4 ml-2" />
            </Link>
            <Link
              to="/dashboard"
              className="inline-flex items-center px-8 py-3.5 font-medium text-white bg-green-700 border-2 border-green-500 hover:bg-green-800 rounded-lg transition-colors"
            >
              View Dashboard
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
