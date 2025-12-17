import { useEffect, useRef, useState } from 'react'
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet'
import { Map as MapIcon } from 'lucide-react'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'

// Fix for default marker icon in React-Leaflet
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'

delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
})

interface LocationSelectorProps {
  latitude: string
  longitude: string
  onLocationChange: (lat: number, lng: number) => void
  onUseCurrentLocation: () => void
  enriching: boolean
}

function LocationMarker({ position, onPositionChange }: { position: [number, number], onPositionChange: (lat: number, lng: number) => void }) {
  useMapEvents({
    click(e) {
      onPositionChange(e.latlng.lat, e.latlng.lng)
    },
  })

  return position ? <Marker position={position} /> : null
}

export function LocationSelector({ latitude, longitude, onLocationChange, onUseCurrentLocation, enriching }: LocationSelectorProps) {
  const [showMap, setShowMap] = useState(false)
  const mapRef = useRef<L.Map>(null)

  const position: [number, number] = [
    parseFloat(latitude) || 5.6037,
    parseFloat(longitude) || -0.1870
  ]

  useEffect(() => {
    if (showMap && mapRef.current && latitude && longitude) {
      mapRef.current.setView(position, 13)
    }
  }, [latitude, longitude, showMap])

  return (
    <div className="space-y-4">
      <div className="bg-gradient-to-r from-blue-50 to-green-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <MapIcon className="w-6 h-6 text-blue-600 mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-bold text-blue-900 mb-2">
              📍 Add Farm Location
            </p>
            <p className="text-xs text-blue-800 mb-3">
              Click "Use My Location" to automatically get your GPS coordinates, or click "Browse Map" to select your farm location visually.
            </p>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={onUseCurrentLocation}
                disabled={enriching}
                className="inline-block px-4 py-2 text-xs font-bold text-white bg-gradient-to-r from-blue-600 to-green-600 rounded-lg hover:from-blue-700 hover:to-green-700 transition disabled:opacity-50"
              >
                {enriching ? 'Capturing Data...' : '📍 Use My Location'}
              </button>
              <button
                type="button"
                onClick={() => setShowMap(!showMap)}
                className="inline-block px-4 py-2 text-xs font-bold text-blue-700 bg-white border-2 border-blue-300 rounded-lg hover:bg-blue-50 transition"
              >
                {showMap ? '📋 Hide Map' : '🗺️ Browse Map'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {showMap && (
        <div className="border border-gray-300 rounded-lg overflow-hidden">
          <div className="h-[400px] relative">
            <MapContainer
              center={position}
              zoom={13}
              style={{ height: '100%', width: '100%' }}
              ref={mapRef}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <LocationMarker position={position} onPositionChange={onLocationChange} />
            </MapContainer>
          </div>
          <div className="bg-gray-50 px-4 py-2 text-xs text-gray-600 border-t border-gray-300">
            💡 Click anywhere on the map to set your farm location. The marker will update automatically.
          </div>
        </div>
      )}
    </div>
  )
}
