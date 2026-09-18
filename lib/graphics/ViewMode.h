// How geometry is drawn. Kept in its own header so that both the render side
// and the parameter store can refer to it without dragging in VSG.
#pragma once

namespace app
{

enum class ViewMode
{
    Facet,     // shaded triangles
    Wireframe, // unique topological edges drawn as lines
    Ray,       // the RayModel's spans drawn as lines
    RayGS,     // the span endpoints drawn as Gaussian splats
    Disk       // span endpoints as hard-edged AA disks (surfels)
};

// True for the modes that need a RayModel to have been cast.
inline bool usesRayModel(ViewMode mode)
{
    return mode == ViewMode::Ray || mode == ViewMode::RayGS || mode == ViewMode::Disk;
}

// True for modes that draw stock from the GaussianSplatCache (Gaussian or disk).
inline bool usesSplatView(ViewMode mode)
{
    return mode == ViewMode::RayGS || mode == ViewMode::Disk;
}

} // namespace app
