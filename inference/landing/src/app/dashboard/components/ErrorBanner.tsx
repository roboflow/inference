import React from 'react';

interface ErrorBannerProps {
  errors: string[];
}

export function ErrorBanner({ errors }: ErrorBannerProps) {
  if (errors.length === 0) {
    return null;
  }

  return (
    <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
      <p className="text-red-800 text-sm">
        {errors.join(" • ")}
      </p>
    </div>
  );
}