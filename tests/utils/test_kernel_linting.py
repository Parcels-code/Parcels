import pytest

from parcels._core.utils.kernel import KernelValidationError, validate_kernel
from parcels._core.warnings import KernelWarning


class TestValidKernel:
    def test_valid_kernel_passes(self):
        @validate_kernel
        def my_kernel(particles, fieldset):
            particles.dx += 1

        assert callable(my_kernel)

    def test_returns_original_function(self):
        def my_kernel(particles, fieldset):
            pass

        result = validate_kernel(my_kernel)
        assert result is my_kernel


# === ERRORS (raise KernelValidationError) ===


class TestSignatureErrors:
    def test_singular_particle_rejected(self):
        with pytest.raises(KernelValidationError, match="signature"):

            @validate_kernel
            def my_kernel(particle, fieldset):
                pass

    def test_extra_time_argument_rejected(self):
        with pytest.raises(KernelValidationError, match="signature"):

            @validate_kernel
            def my_kernel(particles, fieldset, time):
                pass

    def test_wrong_param_names_rejected(self):
        with pytest.raises(KernelValidationError, match="signature"):

            @validate_kernel
            def my_kernel(p, fs):
                pass

    def test_swapped_params_rejected(self):
        with pytest.raises(KernelValidationError, match="signature"):

            @validate_kernel
            def my_kernel(fieldset, particles):
                pass


class TestDeleteErrors:
    def test_particles_delete_rejected(self):
        with pytest.raises(KernelValidationError, match=r"delete.*no longer valid"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particles.delete()

    def test_particle_delete_in_body_rejected(self):
        with pytest.raises(KernelValidationError, match=r"particle\.delete.*no longer valid"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particle = particles[0]
                particle.delete()


class TestDirectLocationAssignmentErrors:
    def test_assign_to_x(self):
        with pytest.raises(KernelValidationError, match="Don't change the location"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particles.x = 1.0

    def test_assign_to_y(self):
        with pytest.raises(KernelValidationError, match="Don't change the location"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particles.y = 1.0

    def test_assign_to_z(self):
        with pytest.raises(KernelValidationError, match="Don't change the location"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particles.z = 1.0

    def test_assign_to_lon(self):
        with pytest.raises(KernelValidationError, match="Don't change the location"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particles.lon = 1.0

    def test_assign_to_lat(self):
        with pytest.raises(KernelValidationError, match="Don't change the location"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particles.lat = 1.0

    def test_assign_to_depth(self):
        with pytest.raises(KernelValidationError, match="Don't change the location"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particles.depth = 1.0

    def test_augassign_to_x(self):
        with pytest.raises(KernelValidationError, match="Don't change the location"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particles.x += 1.0

    def test_augassign_to_y(self):
        with pytest.raises(KernelValidationError, match="Don't change the location"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particles.y += 1.0

    def test_augassign_to_z(self):
        with pytest.raises(KernelValidationError, match="Don't change the location"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particles.z += 1.0

    def test_assign_to_dx_allowed(self):
        @validate_kernel
        def my_kernel(particles, fieldset):
            particles.dx += 1.0

        assert callable(my_kernel)

    def test_assign_to_custom_attr_allowed(self):
        @validate_kernel
        def my_kernel(particles, fieldset):
            particles.temperature = 20.0

        assert callable(my_kernel)


# === WARNINGS (emit KernelWarning) ===


class TestDeprecatedCoordWarnings:
    def test_lon_warns(self):
        with pytest.warns(KernelWarning, match=r"particles\.lon.*particles\.x"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                val = particles.lon  # noqa: F841

    def test_lat_warns(self):
        with pytest.warns(KernelWarning, match=r"particles\.lat.*particles\.y"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                val = particles.lat  # noqa: F841

    def test_depth_warns(self):
        with pytest.warns(KernelWarning, match=r"particles\.depth.*particles\.z"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                val = particles.depth  # noqa: F841


class TestDeprecatedDeltaWarnings:
    def test_particle_dlon_warns(self):
        with pytest.warns(KernelWarning, match=r"particle_dlon.*particles\.dx"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particle_dlon = 0.1  # noqa: F841

    def test_particle_dlat_warns(self):
        with pytest.warns(KernelWarning, match=r"particle_dlat.*particles\.dy"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particle_dlat = 0.1  # noqa: F841

    def test_particle_ddepth_warns(self):
        with pytest.warns(KernelWarning, match=r"particle_ddepth.*particles\.dz"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                particle_ddepth = 0.1  # noqa: F841


class TestIndexSamplingWarnings:
    def test_xi_warns(self):
        with pytest.warns(KernelWarning, match=r"particles\.xi.*before advection"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                val = particles.xi  # noqa: F841

    def test_yi_warns(self):
        with pytest.warns(KernelWarning, match=r"particles\.yi.*before advection"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                val = particles.yi  # noqa: F841

    def test_zi_warns(self):
        with pytest.warns(KernelWarning, match=r"particles\.zi.*before advection"):

            @validate_kernel
            def my_kernel(particles, fieldset):
                val = particles.zi  # noqa: F841


# === MIXED: errors + warnings together ===


class TestMixedViolations:
    def test_warnings_emitted_before_error_raised(self):
        with pytest.warns(KernelWarning, match="lon"):
            with pytest.raises(KernelValidationError) as exc_info:

                @validate_kernel
                def my_kernel(particle, fieldset, time):
                    x = particle.lon  # noqa: F841
                    particle.delete()

        msg = str(exc_info.value)
        assert "signature" in msg
        assert "delete" in msg
        assert "2 validation error(s)" in msg

    def test_multiple_errors_collected(self):
        with pytest.raises(KernelValidationError) as exc_info:

            @validate_kernel
            def my_kernel(particle, fieldset, time):
                particle.delete()
                particle.x = 5

        msg = str(exc_info.value)
        assert "signature" in msg
        assert "delete" in msg
        assert "Don't change the location" in msg
        assert "3 validation error(s)" in msg
